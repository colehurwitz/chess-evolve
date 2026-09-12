"""Tests for pipeline definition — build_position_eval_workflow()."""

from __future__ import annotations

from factory.workflow.primitives import AgentNode, DataNode, Workflow

from chess_evolve.pipeline import GENERATOR_PROMPT, build_position_eval_workflow


class TestBuildPositionEvalWorkflow:
    def test_returns_workflow(self):
        wf = build_position_eval_workflow()
        assert isinstance(wf, Workflow)

    def test_has_positions_data_node(self):
        wf = build_position_eval_workflow()
        assert "positions" in wf.nodes
        assert isinstance(wf.nodes["positions"], DataNode)

    def test_has_generator_agent_node(self):
        wf = build_position_eval_workflow()
        assert "generator" in wf.nodes
        assert isinstance(wf.nodes["generator"], AgentNode)

    def test_generator_has_chess_prompt(self):
        wf = build_position_eval_workflow()
        node = wf.nodes["generator"]
        assert isinstance(node, AgentNode)
        assert node.prompt_template == GENERATOR_PROMPT
        assert "chess" in node.prompt_template.lower()
