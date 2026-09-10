"""Tests for the evolution module."""

from __future__ import annotations

import inspect


class TestSwarmEngineWiring:
    def test_main_is_sync(self):
        from chess_evolve.evolution import main

        assert not inspect.iscoroutinefunction(main)

    def test_main_accepts_project_dir(self):
        from chess_evolve.evolution import main

        sig = inspect.signature(main)
        assert "project_dir" in sig.parameters

    def test_swarm_config_has_frozen_positions(self):
        from factory.outer_loop import SwarmConfig

        config = SwarmConfig(
            benchmark="chess-evolve",
            budget=100,
            population_size=4,
            tournament_size=3,
            frozen_node_ids=["positions"],
            task_module="chess_evolve.tasks:PositionTask",
        )
        assert "positions" in config.frozen_node_ids

    def test_evolution_imports_cleanly(self):
        from chess_evolve.evolution import main  # noqa: F401
