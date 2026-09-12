"""Outer loop: evolutionary prompt optimization via SwarmEngine."""

from __future__ import annotations

from pathlib import Path

from factory.outer_loop import SwarmConfig, SwarmEngine
from factory.outer_loop.evaluator import SwarmEvaluator
from factory.outer_loop.designer import DesignerAgent
from factory.outer_loop.models import EvalResult
from factory.workflow.primitives import AgentNode, Workflow

from chess_evolve.pipeline import build_position_eval_workflow, BUILDER_PROMPT
from chess_evolve.tasks import PositionTask


class ChessEvaluator(SwarmEvaluator):
    """SwarmEvaluator subclass that applies chess-specific patches to all workflows.

    This ensures that researcher/builder nodes are properly configured for chess
    evaluation, even when workflows are created through mutation or other mechanisms
    that don't go through the ChessDesignerAgent.
    """

    def __init__(self, *args, designer=None, **kwargs):
        """Initialize with optional designer for patching."""
        super().__init__(*args, **kwargs)
        self._designer = designer

    def evaluate(
        self,
        workflow: Workflow,
        project_dir: str,
        instances: list[str],
        individual_id: str | None = None,
    ) -> EvalResult:
        """Evaluate a workflow, applying chess patches before evaluation."""
        # Apply chess patches to ensure researcher and builder are properly configured
        if self._designer:
            self._designer._patch_chess_nodes(workflow)
        return super().evaluate(workflow, project_dir, instances, individual_id)


class ChessDesignerAgent(DesignerAgent):
    """Custom designer that adds proper prompt template and reads to builder node."""

    def _patch_chess_nodes(self, workflow):
        """Apply chess-specific patches to researcher and builder nodes.

        This ensures that:
        - Researcher writes analysis to a chess-specific location
        - Builder has the chess-specific prompt template and reads from the right files

        This method is called on every workflow, not just those from design_minimal,
        to ensure patches persist across all workflow iterations and mutations.
        """
        # Patch researcher to write analysis to chess-specific location
        if "researcher" in workflow.nodes:
            researcher = workflow.nodes["researcher"]
            if isinstance(researcher, AgentNode):
                researcher.writes = {".factory/chess/analysis.md"}

        # Patch the builder node to include the proper prompt template and reads
        if "builder" in workflow.nodes:
            builder = workflow.nodes["builder"]
            if isinstance(builder, AgentNode):
                builder.prompt_template = BUILDER_PROMPT
                # Ensure builder reads the researcher's output and board state
                builder.reads = {
                    ".factory/chess/board_state.md",
                    ".factory/chess/analysis.md",
                    ".factory/chess/memory.md",
                }

        return workflow

    def design_minimal(self, benchmark_spec, seed_workflow=None, frozen_node_ids=None):
        """Create minimal workflow with chess-specific builder prompt and configuration."""
        wf = super().design_minimal(benchmark_spec)
        return self._patch_chess_nodes(wf)

    def design_thorough(self, benchmark_spec, seed_workflow=None, frozen_node_ids=None):
        """Create thorough workflow with chess-specific patches."""
        wf = super().design_thorough(benchmark_spec)
        return self._patch_chess_nodes(wf)

    def design_custom(self, benchmark_spec, hints=None):
        """Create custom workflow with chess-specific patches."""
        wf = super().design_custom(benchmark_spec, hints=hints)
        return self._patch_chess_nodes(wf)


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

    designer = ChessDesignerAgent()

    evaluator = ChessEvaluator(
        config,
        designer=designer,
        inner_loop_factory=True,
        project_dir=project_dir,
    )
    engine = SwarmEngine(config, evaluator, designer=designer, project_dir=project_dir)
    result = engine.run(workflow, project_dir=str(project_dir))

    print(f"Best score: {result.best_score}")
    print(f"Generations: {result.generations_completed}")
    print(f"Convergence: {result.convergence_reason}")
