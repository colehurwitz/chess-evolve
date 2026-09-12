"""Outer loop: evolutionary prompt optimization via SwarmEngine."""

from __future__ import annotations

from pathlib import Path

from factory.outer_loop import SwarmConfig, SwarmEngine
from factory.outer_loop.evaluator import SwarmEvaluator

from chess_evolve.pipeline import build_position_eval_workflow
from chess_evolve.tasks import PositionTask


def main(
    project_dir: Path | None = None, task_type: str = "position",
) -> None:
    """Run chess evolution via SwarmEngine."""
    project_dir = project_dir or Path.cwd()

    if task_type == "game":
        from chess_evolve.pipeline import build_game_eval_workflow
        from chess_evolve.tasks import GameTask

        task = GameTask()
        workflow = build_game_eval_workflow()
        task_module = "chess_evolve.tasks:GameTask"
        frozen = ["games", "builder"]
    else:
        task = PositionTask()
        workflow = build_position_eval_workflow()
        task_module = "chess_evolve.tasks:PositionTask"
        frozen = ["positions"]

    config = SwarmConfig(
        benchmark="chess-evolve",
        budget=100,
        population_size=4,
        tournament_size=3,
        frozen_node_ids=frozen,
        task_module=task_module,
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
