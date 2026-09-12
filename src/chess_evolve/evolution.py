"""Outer loop: evolutionary prompt optimization via SwarmEngine."""

from __future__ import annotations

from pathlib import Path

from factory.outer_loop import SwarmConfig, SwarmEngine
from factory.outer_loop.evaluator import SwarmEvaluator
from factory.outer_loop.designer import DesignerAgent
from factory.workflow.primitives import AgentNode

from chess_evolve.pipeline import build_position_eval_workflow, BUILDER_PROMPT
from chess_evolve.tasks import PositionTask


class ChessDesignerAgent(DesignerAgent):
    """Custom designer that adds proper prompt template and reads to builder node."""

    def design_minimal(self, benchmark_spec, seed_workflow=None, frozen_node_ids=None):
        """Create minimal workflow with chess-specific builder prompt and configuration."""
        wf = super().design_minimal(benchmark_spec)

        # Patch the builder node to include the proper prompt template and reads
        if "builder" in wf.nodes:
            builder = wf.nodes["builder"]
            if isinstance(builder, AgentNode):
                builder.prompt_template = BUILDER_PROMPT
                # Ensure builder reads the researcher's output
                builder.reads = {
                    ".factory/chess/board_state.md",
                    ".factory/strategy/research.md",
                    ".factory/chess/memory.md",
                }

        return wf


def main(
    project_dir: Path | None = None, task_type: str = "position",
) -> None:
    """Run chess evolution via SwarmEngine."""
    project_dir = project_dir or Path.cwd()

    if task_type == "game":
        import os

        from factory.workflow.registry import WorkflowRegistry

        from chess_evolve.tasks import GameTask

        task = GameTask()
        workflow = WorkflowRegistry.get_workflow("chess-game", project_dir)
        if workflow is None:
            raise RuntimeError(
                "chess-game workflow not found in .factory/workflows/. "
                "Expected .factory/workflows/chess_game.py with "
                'meta["name"]="chess-game".'
            )
        model_override = os.environ.get("ANTHROPIC_MODEL")
        if model_override and "generator" in workflow.nodes:
            workflow.nodes["generator"].model = model_override
        task_module = "chess_evolve.tasks:GameTask"
        frozen = ["games"]
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
    designer = ChessDesignerAgent()
    engine = SwarmEngine(config, evaluator, designer=designer, project_dir=project_dir)
    result = engine.run(workflow, project_dir=str(project_dir))

    print(f"Best score: {result.best_score}")
    print(f"Generations: {result.generations_completed}")
    print(f"Convergence: {result.convergence_reason}")
