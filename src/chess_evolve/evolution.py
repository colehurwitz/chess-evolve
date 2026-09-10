"""Outer loop: thin wrapper around factory's SwarmEngine.

chess-evolve owns only the Task (PositionTask), the Workflow
(build_position_eval_workflow), and the evaluator adapter
(ChessSwarmEvaluator).  Everything else — population management,
mutation, reflection, archive, plateau detection — is delegated to
SwarmEngine.
"""

from __future__ import annotations

from pathlib import Path

from factory.outer_loop import SwarmConfig, SwarmEngine
from factory.outer_loop.mutations import WeightedRandomStrategy
from factory.workflow.primitives import Workflow

from chess_evolve.evaluator import ChessSwarmEvaluator
from chess_evolve.pipeline import PipelineConfig, build_position_eval_workflow


def main() -> None:
    """Run the evolutionary search via SwarmEngine."""
    project_dir = Path.cwd()

    # 1. Build seed workflow (DataNode + PositionTask)
    seed_wf = build_position_eval_workflow(PipelineConfig())

    # 2. Configure SwarmEngine
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

    # 3. Create evaluator (chess-evolve's only adapter)
    evaluator = ChessSwarmEvaluator(project_dir, config)

    # 4. Chess-tuned mutation weights
    strategy = WeightedRandomStrategy(
        weights={
            "knob_mutate": 0.35,
            "prompt_mutate": 0.30,
            "node_insert": 0.15,
            "node_remove": 0.10,
            "param_mutate": 0.10,
        },
        mutation_rate=0.3,
    )

    # 5. Run — SwarmEngine handles everything:
    #    seed population, mutation, evaluation, reflection,
    #    archive management, plateau detection, termination
    engine = SwarmEngine(
        config=config,
        evaluator=evaluator,
        strategy=strategy,
        project_dir=project_dir,
    )
    result = engine.run(seed_wf, project_dir=str(project_dir))

    # 6. Report results
    print(f"Best score: {result.best_score}")
    print(f"Total cost: ${result.total_cost_usd:.2f}")
    print(f"Generations: {result.generations_completed}")
    print(f"Evaluations: {result.total_evaluations}")
    print(f"Archive size: {result.archive_size}")
    print(f"Convergence: {result.convergence_reason}")
    if result.best_workflow_data:
        best_wf = Workflow.from_dict(result.best_workflow_data)  # type: ignore[arg-type]
        print(f"Best workflow knobs: {best_wf.knob_values}")
