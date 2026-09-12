"""Tests for the chess-game workflow loaded via WorkflowRegistry."""

from __future__ import annotations

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
