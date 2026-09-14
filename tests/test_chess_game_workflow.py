"""Tests for the chess-game workflow loaded via WorkflowRegistry."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from factory.workflow.primitives import DataNode, GateNode, Workflow
from factory.workflow.registry import WorkflowRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestRegistryDiscovery:
    def test_registry_discovers_chess_game(self):
        wf = WorkflowRegistry.get_workflow("chess-game", PROJECT_ROOT)
        assert wf is not None
        assert isinstance(wf, Workflow)

    def test_workflow_has_expected_nodes(self):
        wf = WorkflowRegistry.get_workflow("chess-game", PROJECT_ROOT)
        assert wf is not None
        assert "games" in wf.nodes
        assert "generator" in wf.nodes
        assert "game_gate" in wf.nodes

    def test_workflow_start_node_is_games(self):
        wf = WorkflowRegistry.get_workflow("chess-game", PROJECT_ROOT)
        assert wf is not None
        assert wf.start_node == "games"


class TestWorkflowNodeStructure:
    def test_games_is_data_node(self):
        wf = WorkflowRegistry.get_workflow("chess-game", PROJECT_ROOT)
        assert wf is not None
        games = wf.nodes["games"]
        assert isinstance(games, DataNode)

    def test_games_task_ref_contains_game_task(self):
        wf = WorkflowRegistry.get_workflow("chess-game", PROJECT_ROOT)
        assert wf is not None
        games = wf.nodes["games"]
        assert isinstance(games, DataNode)
        assert "GameTask" in games.task_ref

    def test_games_has_subgraph_entry_and_exit(self):
        wf = WorkflowRegistry.get_workflow("chess-game", PROJECT_ROOT)
        assert wf is not None
        games = wf.nodes["games"]
        assert isinstance(games, DataNode)
        assert games.subgraph_entry is not None
        assert games.subgraph_exit is not None
        assert games.subgraph_exit == "exit_game-loop"

    def test_workflow_node_structure(self):
        """Verify 'games' is a DataNode with task_ref, subgraph_entry, subgraph_exit."""
        wf = WorkflowRegistry.get_workflow("chess-game", PROJECT_ROOT)
        assert wf is not None
        games = wf.nodes["games"]
        assert isinstance(games, DataNode)
        assert "GameTask" in games.task_ref
        assert games.subgraph_entry is not None
        assert games.subgraph_exit == "exit_game-loop"


class TestGameGate:
    def test_game_gate_uses_sys_executable(self):
        wf = WorkflowRegistry.get_workflow("chess-game", PROJECT_ROOT)
        assert wf is not None
        gate = wf.nodes["game_gate"]
        assert isinstance(gate, GateNode)
        assert sys.executable in gate.evaluator_command
        assert "python3" not in gate.evaluator_command


class TestAnthropicModelOverride:
    def test_anthropic_model_env_override(self, monkeypatch):
        """Simulate evolution.py's game branch: ANTHROPIC_MODEL sets generator model."""
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

        import os

        wf = WorkflowRegistry.get_workflow("chess-game", PROJECT_ROOT)
        assert wf is not None

        # Reproduce the workaround from evolution.py
        model_override = os.environ.get("ANTHROPIC_MODEL")
        if model_override and "generator" in wf.nodes:
            wf.nodes["generator"].model = model_override

        assert wf.nodes["generator"].model == "claude-haiku-4-5-20251001"


class TestDeprecatedPipeline:
    def test_deprecated_pipeline_warns(self):
        from chess_evolve.pipeline import build_game_eval_workflow

        with pytest.warns(DeprecationWarning, match="deprecated"):
            build_game_eval_workflow()


def _load_chess_game_module():
    """Load the chess_game workflow module for testing."""
    spec = importlib.util.spec_from_file_location(
        "chess_game_wf",
        PROJECT_ROOT / ".factory" / "workflows" / "chess_game.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestBuildGeneratorPrompt:
    """Unit tests for the build_generator_prompt helper."""

    @pytest.fixture(autouse=True)
    def _load_module(self):
        self.mod = _load_chess_game_module()

    def test_default_prompt_contains_balanced_style(self):
        prompt = self.mod.build_generator_prompt()
        assert "balanced" in prompt
        assert "Output ONLY the UCI move" in prompt

    def test_tactical_style(self):
        prompt = self.mod.build_generator_prompt(style="tactical")
        assert "tactical" in prompt

    def test_detailed_verbosity(self):
        prompt = self.mod.build_generator_prompt(verbosity="detailed")
        assert "explain your reasoning" in prompt
        assert "Output ONLY" not in prompt

    def test_brief_reason_verbosity(self):
        prompt = self.mod.build_generator_prompt(verbosity="brief_reason")
        assert "one-sentence reason" in prompt

    def test_unknown_verbosity_falls_back_to_uci_only(self):
        prompt = self.mod.build_generator_prompt(verbosity="unknown_level")
        assert "Output ONLY the UCI move" in prompt


class TestOptKnobs:
    """Tests for OptKnob declarations and compile() propagation."""

    def test_knob_values_populated_after_compile(self):
        """knob_values dict has 'style' and 'verbosity' after compile()."""
        wf = WorkflowRegistry.get_workflow("chess-game", PROJECT_ROOT)
        assert wf is not None
        assert "style" in wf.knob_values
        assert "verbosity" in wf.knob_values

    def test_knob_defaults(self):
        """Default values match the knob declarations."""
        wf = WorkflowRegistry.get_workflow("chess-game", PROJECT_ROOT)
        assert wf is not None
        assert wf.knob_values["style"] == "balanced"
        assert wf.knob_values["verbosity"] == "uci_only"

    def test_knob_bounds_populated(self):
        """knob_bounds dict has entries for both knobs."""
        wf = WorkflowRegistry.get_workflow("chess-game", PROJECT_ROOT)
        assert wf is not None
        assert "style" in wf.knob_bounds
        assert "verbosity" in wf.knob_bounds
        assert wf.knob_bounds["style"] == [
            "balanced", "tactical", "positional", "aggressive",
        ]
        assert wf.knob_bounds["verbosity"] == [
            "uci_only", "brief_reason", "detailed",
        ]

    def test_knob_expandable_style_present(self):
        """style is expandable — should appear in knob_expandable."""
        wf = WorkflowRegistry.get_workflow("chess-game", PROJECT_ROOT)
        assert wf is not None
        assert "style" in wf.knob_expandable
        assert wf.knob_expandable["style"]  # non-empty hint string

    def test_knob_expandable_verbosity_absent(self):
        """verbosity is NOT expandable — should NOT appear in knob_expandable."""
        wf = WorkflowRegistry.get_workflow("chess-game", PROJECT_ROOT)
        assert wf is not None
        assert "verbosity" not in wf.knob_expandable

    def test_generator_prompt_uses_default_knob_values(self):
        """The generator node's prompt_template reflects default knob values."""
        wf = WorkflowRegistry.get_workflow("chess-game", PROJECT_ROOT)
        assert wf is not None
        generator = wf.nodes["generator"]
        assert "balanced" in generator.prompt_template
        assert "Output ONLY the UCI move" in generator.prompt_template


class TestKnobMutateIntegration:
    """Integration tests: KNOB_MUTATE operator works with the workflow's knobs."""

    def test_mutate_knob_returns_result(self):
        """mutate_knob() returns a non-None result (mutation succeeded)."""
        from factory.outer_loop.mutations import mutate_knob

        wf = WorkflowRegistry.get_workflow("chess-game", PROJECT_ROOT)
        assert wf is not None
        # Run multiple attempts — mutate_knob has randomness
        found = False
        for _ in range(20):
            result = mutate_knob(wf, expander=None)
            if result is not None:
                found = True
                break
        assert found, "mutate_knob() never returned a result in 20 attempts"

    def test_mutate_knob_changes_a_value(self):
        """After mutation, at least one knob_value differs from the default."""
        from factory.outer_loop.mutations import mutate_knob

        wf = WorkflowRegistry.get_workflow("chess-game", PROJECT_ROOT)
        assert wf is not None
        original_values = dict(wf.knob_values)
        for _ in range(20):
            result = mutate_knob(wf, expander=None)
            if result is not None:
                mutated_wf, record = result
                assert mutated_wf.knob_values != original_values
                break

    def test_mutate_knob_targets_declared_knobs(self):
        """Mutation targets 'style' or 'verbosity', not synthetic _prompt_ knobs."""
        from factory.outer_loop.mutations import mutate_knob

        wf = WorkflowRegistry.get_workflow("chess-game", PROJECT_ROOT)
        assert wf is not None
        for _ in range(20):
            result = mutate_knob(wf, expander=None)
            if result is not None:
                _, record = result
                # record.description or record fields indicate which knob changed
                # The mutated value must be for 'style' or 'verbosity'
                changed_keys = {
                    k for k in result[0].knob_values
                    if result[0].knob_values[k] != wf.knob_values.get(k)
                }
                assert changed_keys & {"style", "verbosity"}, (
                    f"Mutation changed unexpected keys: {changed_keys}"
                )
                break
