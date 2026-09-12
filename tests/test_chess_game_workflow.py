"""Tests for the chess-game project-local workflow.

Hermetic: no live LLM, no live Stockfish — tests only verify workflow
structure, node types, graph connectivity, and metadata.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from factory.workflow.primitives import AgentNode, DataNode, GateNode, Workflow

# ── Helper: import the project-local workflow file ──────────────


def _load_chess_game_module():
    """Import .factory/workflows/chess_game.py as a module.

    Mirrors how the factory registry discovers project-local workflows:
    importlib.util.spec_from_file_location + exec_module.
    """
    wf_path = (
        Path(__file__).resolve().parent.parent
        / ".factory"
        / "workflows"
        / "chess_game.py"
    )
    if not wf_path.exists():
        pytest.skip(f"Workflow file not found: {wf_path}")
    spec = importlib.util.spec_from_file_location("chess_game_wf", str(wf_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ── TestMeta ────────────────────────────────────────────────────


class TestMeta:
    """Verify the meta dict required by the factory registry."""

    def test_meta_has_name(self) -> None:
        mod = _load_chess_game_module()
        assert hasattr(mod, "meta")
        assert mod.meta["name"] == "chess-game"

    def test_meta_has_description(self) -> None:
        mod = _load_chess_game_module()
        assert "description" in mod.meta
        assert len(mod.meta["description"]) > 0


# ── TestWorkflowStructure ──────────────────────────────────────


class TestWorkflowStructure:
    """Verify workflow() returns a valid Workflow with correct nodes."""

    def test_returns_workflow_instance(self) -> None:
        mod = _load_chess_game_module()
        wf = mod.workflow()
        assert isinstance(wf, Workflow)

    def test_workflow_name_matches_meta(self) -> None:
        mod = _load_chess_game_module()
        wf = mod.workflow()
        assert wf.name == "chess-game"

    def test_start_node_is_games(self) -> None:
        mod = _load_chess_game_module()
        wf = mod.workflow()
        assert wf.start_node == "games"

    def test_has_data_node_games(self) -> None:
        mod = _load_chess_game_module()
        wf = mod.workflow()
        assert "games" in wf.nodes
        assert isinstance(wf.nodes["games"], DataNode)

    def test_data_node_task_ref(self) -> None:
        mod = _load_chess_game_module()
        wf = mod.workflow()
        data_node = wf.nodes["games"]
        assert data_node.task_ref == "chess_evolve.tasks:GameTask"

    def test_data_node_parallelism(self) -> None:
        mod = _load_chess_game_module()
        wf = mod.workflow()
        data_node = wf.nodes["games"]
        assert data_node.parallelism == 1

    def test_has_generator_agent_node(self) -> None:
        mod = _load_chess_game_module()
        wf = mod.workflow()
        gen_nodes = [
            nid
            for nid, n in wf.nodes.items()
            if isinstance(n, AgentNode) and "generator" in nid
        ]
        assert len(gen_nodes) >= 1, (
            f"No AgentNode with 'generator' in id. Nodes: {list(wf.nodes.keys())}"
        )

    def test_has_game_gate_node(self) -> None:
        mod = _load_chess_game_module()
        wf = mod.workflow()
        gate_nodes = [
            nid
            for nid, n in wf.nodes.items()
            if isinstance(n, GateNode) and "game_gate" in nid
        ]
        assert len(gate_nodes) >= 1, (
            f"No GateNode with 'game_gate' in id. Nodes: {list(wf.nodes.keys())}"
        )

    def test_game_gate_is_fn_type(self) -> None:
        mod = _load_chess_game_module()
        wf = mod.workflow()
        gate_nodes = {
            nid: n
            for nid, n in wf.nodes.items()
            if isinstance(n, GateNode) and "game_gate" in nid
        }
        for _nid, gate in gate_nodes.items():
            assert gate.evaluator_type == "fn"
            assert "chess_game_gate" in (gate.evaluator_command or "")

    def test_has_exit_node(self) -> None:
        mod = _load_chess_game_module()
        wf = mod.workflow()
        exit_nodes = [
            nid for nid in wf.nodes if "exit" in nid and "game-loop" in nid
        ]
        assert len(exit_nodes) >= 1, (
            f"No exit_game-loop node. Nodes: {list(wf.nodes.keys())}"
        )


# ── TestSubgraphConnectivity ───────────────────────────────────


class TestSubgraphConnectivity:
    """Verify DataNode subgraph_entry and subgraph_exit reference valid nodes."""

    def test_subgraph_entry_exists(self) -> None:
        mod = _load_chess_game_module()
        wf = mod.workflow()
        data_node = wf.nodes["games"]
        assert data_node.subgraph_entry in wf.nodes, (
            f"subgraph_entry={data_node.subgraph_entry!r} "
            f"not in nodes: {list(wf.nodes.keys())}"
        )

    def test_subgraph_exit_exists(self) -> None:
        mod = _load_chess_game_module()
        wf = mod.workflow()
        data_node = wf.nodes["games"]
        assert data_node.subgraph_exit in wf.nodes, (
            f"subgraph_exit={data_node.subgraph_exit!r} "
            f"not in nodes: {list(wf.nodes.keys())}"
        )

    def test_subgraph_entry_is_not_data_node(self) -> None:
        mod = _load_chess_game_module()
        wf = mod.workflow()
        data_node = wf.nodes["games"]
        entry_node = wf.nodes[data_node.subgraph_entry]
        assert not isinstance(entry_node, DataNode)

    def test_subgraph_exit_is_exit_node(self) -> None:
        mod = _load_chess_game_module()
        wf = mod.workflow()
        data_node = wf.nodes["games"]
        assert data_node.subgraph_exit == "exit_game-loop"


# ── TestGraphValidation ────────────────────────────────────────


class TestGraphValidation:
    """Verify the workflow passes structural validation."""

    def test_validate_graph_no_critical_issues(self) -> None:
        """validate_graph() should return no issues (or only known benign ones).

        The reads/writes warning about board_state.md and memory.md is expected
        because GameTask.setup() creates those files — not a predecessor node.
        """
        mod = _load_chess_game_module()
        wf = mod.workflow()
        issues = wf.validate_graph()
        # Filter out the known benign reads/writes warning
        critical = [
            i
            for i in issues
            if "reads" not in i or "board_state" not in i
        ]
        assert critical == [], f"Unexpected validation issues: {critical}"


# ── TestGeneratorConfiguration ─────────────────────────────────


class TestGeneratorConfiguration:
    """Verify the generator AgentNode is configured correctly."""

    def test_generator_reads_board_state(self) -> None:
        mod = _load_chess_game_module()
        wf = mod.workflow()
        gen_nodes = [
            n
            for nid, n in wf.nodes.items()
            if isinstance(n, AgentNode) and "generator" in nid
        ]
        gen = gen_nodes[0]
        assert ".factory/chess/board_state.md" in gen.reads

    def test_generator_reads_memory(self) -> None:
        mod = _load_chess_game_module()
        wf = mod.workflow()
        gen_nodes = [
            n
            for nid, n in wf.nodes.items()
            if isinstance(n, AgentNode) and "generator" in nid
        ]
        gen = gen_nodes[0]
        assert ".factory/chess/memory.md" in gen.reads

    def test_generator_writes_move(self) -> None:
        mod = _load_chess_game_module()
        wf = mod.workflow()
        gen_nodes = [
            n
            for nid, n in wf.nodes.items()
            if isinstance(n, AgentNode) and "generator" in nid
        ]
        gen = gen_nodes[0]
        assert ".factory/chess/move.md" in gen.writes


# ── TestComposeIntegration ──────────────────────────────────────


class TestComposeIntegration:
    """Verify compose(workflow, GameTask) works end-to-end."""

    def test_compose_with_game_task(self, tmp_path: Path) -> None:
        from factory.compose import compose
        from factory.inner_loop import InnerLoop

        from chess_evolve.tasks import GameTask

        mod = _load_chess_game_module()
        wf = mod.workflow()
        task = GameTask()
        loop = compose(wf, task, tmp_path)
        # compose() should return an InnerLoop
        assert isinstance(loop, InnerLoop)
        assert loop.mode == "chess-game"


# ── TestGateScript ──────────────────────────────────────────────


class TestGateScript:
    """Verify the standalone gate script exists and is importable."""

    def test_gate_script_exists(self) -> None:
        gate_path = (
            Path(__file__).resolve().parent.parent
            / ".factory"
            / "workflows"
            / "chess_game_gate.py"
        )
        assert gate_path.exists(), f"Gate script not found: {gate_path}"

    def test_gate_script_has_advance_game_state(self) -> None:
        gate_path = (
            Path(__file__).resolve().parent.parent
            / ".factory"
            / "workflows"
            / "chess_game_gate.py"
        )
        content = gate_path.read_text()
        assert "def advance_game_state" in content
        assert "PROCEED" in content
        assert "RELOOP" in content
