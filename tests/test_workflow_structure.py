"""Structural invariant tests for chess-game and position-eval workflows.

Tests cover two production bugs:
1. Subgraph node freezing — validate_and_repair() prunes subgraph nodes
   because it doesn't add DataNode->subgraph_entry implicit edges.
2. _resolve_project_root() — workspace.parent.parent is wrong for eval worktrees.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from factory.outer_loop.mutations import remove_node, validate_and_repair
from factory.workflow.primitives import DataNode

from chess_evolve.pipeline import build_position_eval_workflow, compute_frozen_nodes
from chess_evolve.tasks import _resolve_project_root

# ── Helper ──────────────────────────────────────────────────────────


def _load_chess_game_workflow():
    """Load the chess-game workflow via the registry."""
    from factory.workflow.registry import WorkflowRegistry

    wf = WorkflowRegistry.get_workflow("chess-game", Path("."))
    assert wf is not None, "chess-game workflow not found in registry"
    return wf


# ── Subgraph freezing tests ─────────────────────────────────────────


GAME_SUBGRAPH_NODES = {"generator", "game_gate", "exit_game-loop"}


class TestSubgraphFreezing:
    """Tests for the subgraph node freezing fix (Bug 1)."""

    def test_frozen_list_includes_all_subgraph_nodes(self):
        """compute_frozen_nodes returns games + all subgraph nodes."""
        wf = _load_chess_game_workflow()
        frozen = compute_frozen_nodes(wf, "games")
        assert "games" in frozen
        for node_id in GAME_SUBGRAPH_NODES:
            assert node_id in frozen, f"{node_id} missing from frozen list"

    def test_subgraph_nodes_survive_validate_and_repair(self):
        """With all subgraph nodes frozen, no mutation can remove them,
        so the workflow structure remains intact through validate_and_repair.

        We verify that when we DON'T touch the subgraph nodes (simulating
        correct freezing), validate_and_repair preserves them only if
        DataNode implicit edges are present. Since they're NOT present,
        this test documents that freezing is the correct workaround:
        frozen nodes aren't mutated, so the full subgraph stays in place
        and the loop edges keep them connected to each other.
        """
        wf = _load_chess_game_workflow()
        frozen = set(compute_frozen_nodes(wf, "games"))

        # Verify remove_node returns None for every frozen node
        for node_id in frozen:
            result = remove_node(wf, node_id, frozen_nodes=frozen)
            assert result is None, (
                f"remove_node should return None for frozen node {node_id}"
            )

    def test_verify_chain_intact_after_freezing(self):
        """DataNode still has task_ref and subgraph_entry pointing to
        an existing node after mutations are blocked by freezing."""
        wf = _load_chess_game_workflow()
        frozen = set(compute_frozen_nodes(wf, "games"))

        # Confirm no frozen node can be removed
        for node_id in frozen:
            assert remove_node(wf, node_id, frozen_nodes=frozen) is None

        # DataNode structural integrity
        data_node = wf.nodes["games"]
        assert isinstance(data_node, DataNode)
        assert data_node.task_ref == "chess_evolve.tasks:GameTask"
        assert data_node.subgraph_entry in wf.nodes, (
            f"subgraph_entry '{data_node.subgraph_entry}' not in workflow nodes"
        )
        assert data_node.subgraph_exit in wf.nodes, (
            f"subgraph_exit '{data_node.subgraph_exit}' not in workflow nodes"
        )

    def test_validate_and_repair_prunes_without_freezing(self):
        """Demonstrate the bug: without DataNode implicit edges,
        validate_and_repair prunes subgraph nodes as unreachable."""
        wf = _load_chess_game_workflow()

        # Before repair: all subgraph nodes present
        for node_id in GAME_SUBGRAPH_NODES:
            assert node_id in wf.nodes

        # validate_and_repair does NOT add DataNode->subgraph_entry edges,
        # so subgraph nodes appear unreachable from start_node ("games")
        repaired = validate_and_repair(wf)

        if repaired is not None:
            # At least one subgraph node should be pruned (the bug)
            pruned = GAME_SUBGRAPH_NODES - set(repaired.nodes.keys())
            assert len(pruned) > 0, (
                "Expected subgraph nodes to be pruned without freezing — "
                "has mutations.py been fixed upstream?"
            )

    def test_position_workflow_frozen_nodes(self):
        """compute_frozen_nodes for position workflow includes positions + generator."""
        wf = build_position_eval_workflow()
        frozen = compute_frozen_nodes(wf, "positions")
        assert "positions" in frozen
        assert "generator" in frozen
        assert len(frozen) == 2


# ── Path resolution tests ───────────────────────────────────────────


class TestResolveProjectRoot:
    """Tests for _resolve_project_root() (Bug 2)."""

    def test_resolve_project_root_in_worktree(self):
        """In a git worktree, _resolve_project_root returns the main repo root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            main_repo = Path(tmpdir) / "main-repo"
            main_repo.mkdir()

            # Initialize main repo
            subprocess.run(
                ["git", "init", str(main_repo)],
                capture_output=True, check=True,
            )
            subprocess.run(
                ["git", "-C", str(main_repo), "commit",
                 "--allow-empty", "-m", "init"],
                capture_output=True, check=True,
            )

            # Create a worktree (simulating eval worktree)
            wt_path = Path(tmpdir) / "eval-worktrees" / "wt-abc123"
            wt_path.parent.mkdir(parents=True)
            subprocess.run(
                ["git", "-C", str(main_repo), "worktree", "add",
                 "-b", "eval-branch", str(wt_path)],
                capture_output=True, check=True,
            )

            # _resolve_project_root should return the main repo, not parent.parent
            resolved = _resolve_project_root(wt_path)
            assert resolved == main_repo, (
                f"Expected {main_repo}, got {resolved}. "
                f"parent.parent would give {wt_path.parent.parent}"
            )
            # Verify parent.parent gives the WRONG answer
            assert wt_path.parent.parent != main_repo

    def test_resolve_project_root_in_normal_repo(self):
        """In a non-worktree checkout, _resolve_project_root returns
        the repo root (same as what parent-based resolution would give
        for a workspace at repo/.factory/something/)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir) / "my-repo"
            repo.mkdir()
            subprocess.run(
                ["git", "init", str(repo)],
                capture_output=True, check=True,
            )

            # _resolve_project_root called from the repo root itself
            resolved = _resolve_project_root(repo)
            assert resolved == repo, f"Expected {repo}, got {resolved}"

    def test_resolve_project_root_non_git_fallback(self):
        """In a non-git directory, _resolve_project_root falls back
        to returning the workspace itself."""
        with tempfile.TemporaryDirectory() as tmpdir:
            non_git = Path(tmpdir) / "not-a-repo"
            non_git.mkdir()
            resolved = _resolve_project_root(non_git)
            assert resolved == non_git
