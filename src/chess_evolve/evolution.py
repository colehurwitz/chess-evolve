"""Outer loop: evolutionary prompt optimization via SwarmEngine."""

from __future__ import annotations

from pathlib import Path

from factory.outer_loop import SwarmConfig, SwarmEngine
from factory.outer_loop.evaluator import SwarmEvaluator

from chess_evolve.pipeline import build_position_eval_workflow
from chess_evolve.tasks import PositionTask


def main(project_dir: Path | None = None) -> None:
    """Run chess evolution via SwarmEngine."""
    project_dir = project_dir or Path.cwd()

    task = PositionTask()
    workflow = build_position_eval_workflow()

    config = SwarmConfig(
        benchmark="chess-evolve",
        budget=100,
        population_size=4,
        tournament_size=3,
        frozen_node_ids=["positions"],
        task_module="chess_evolve.tasks:PositionTask",
    )
    config.set_task(task)

    evaluator = SwarmEvaluator(
        config,
        inner_loop_factory=True,
        project_dir=project_dir,
    )
    engine = SwarmEngine(config, evaluator, project_dir=project_dir)
    result = engine.run(workflow, project_dir=str(project_dir))

    print(f"Best score: {result.best_score}")
    print(f"Generations: {result.generations_completed}")
    print(f"Convergence: {result.convergence_reason}")
