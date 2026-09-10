"""Tests for the evolution loop wiring (SwarmEngine integration)."""

from __future__ import annotations

from unittest.mock import MagicMock

from factory.outer_loop import SwarmConfig, SwarmEngine
from factory.outer_loop.engine import OuterLoopResult
from factory.outer_loop.evaluator import EvalResult

from chess_evolve.pipeline import (
    PipelineConfig,
    build_pipeline,
    build_position_eval_workflow,
)


class TestBuildAndCompile:
    def test_default_pipeline_compiles(self):
        wf = build_pipeline().compile()
        assert "generator" in wf.nodes
        assert len(wf.nodes) >= 2

    def test_different_configs_produce_different_workflows(self):
        wf1 = build_pipeline(PipelineConfig(max_retries=1)).compile()
        wf2 = build_pipeline(PipelineConfig(max_retries=5)).compile()
        assert wf1.knob_values != wf2.knob_values


class TestSwarmEngineWiring:
    def test_swarm_config_valid(self):
        """Construct SwarmConfig with chess-evolve parameters without errors."""
        config = SwarmConfig(
            benchmark="chess-evolve",
            budget=100,
            population_size=4,
            tournament_size=3,
            mutation_rate=0.3,
            target_score=0.8,
            frozen_node_ids=["positions"],
            task_module="chess_evolve.tasks:PositionTask",
            plateau_window=3,
            plateau_threshold=0.01,
        )
        assert config.benchmark == "chess-evolve"
        assert config.frozen_node_ids == ["positions"]
        assert config.population_size == 4

    def test_seed_workflow_has_data_node(self):
        """The seed workflow from build_position_eval_workflow has the DataNode."""
        wf = build_position_eval_workflow(PipelineConfig())
        assert "positions" in wf.nodes
        assert "generator" in wf.nodes
        data_node = wf.nodes["positions"]
        assert data_node.task_ref == "chess_evolve.tasks:PositionTask"

    def test_engine_run_with_mocked_evaluator(self, tmp_path):
        """SwarmEngine.run() with a mocked evaluator produces an OuterLoopResult."""
        config = SwarmConfig(
            benchmark="chess-evolve",
            budget=2,
            population_size=2,
            tournament_size=2,
            mutation_rate=0.3,
            frozen_node_ids=["positions"],
            task_module="chess_evolve.tasks:PositionTask",
            designer_count=0,
        )
        seed_wf = build_position_eval_workflow(PipelineConfig())

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = EvalResult(
            score=0.5, cost_usd=0.01, benchmark_score=0.5,
        )
        mock_evaluator.get_cycle_record.return_value = None
        mock_evaluator.cache = MagicMock()
        mock_evaluator.checkpoint_cache = MagicMock()

        result = SwarmEngine(
            config=config,
            evaluator=mock_evaluator,
            project_dir=tmp_path,
        ).run(seed_wf, project_dir=str(tmp_path))

        assert isinstance(result, OuterLoopResult)
        assert result.best_score >= 0.0

    def test_frozen_node_preserved_in_config(self):
        """The 'positions' DataNode ID appears in frozen_node_ids."""
        config = SwarmConfig(
            benchmark="chess-evolve",
            budget=10,
            population_size=4,
            frozen_node_ids=["positions"],
            task_module="chess_evolve.tasks:PositionTask",
        )
        assert "positions" in config.frozen_node_ids

    def test_evolution_main_is_sync(self):
        """evolution.main() is a sync function (not async)."""
        import inspect

        from chess_evolve.evolution import main

        assert not inspect.iscoroutinefunction(main)
