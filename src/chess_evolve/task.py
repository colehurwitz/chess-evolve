"""Factory Task abstraction for chess-evolve.

Wraps the existing evaluation harness (evaluate_pipeline, build_pipeline,
setup_workspace) behind factory's four-hook Task interface so the factory
outer loop can drive chess evaluation via Task.instances / setup / verify.

The outer loop hands each candidate workflow to Task.run(); its knob_values
are what distinguish one candidate from another, so they are translated back
into a PipelineConfig before the pipeline is built.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Iterator

from factory.task import Task, TaskConstraints, TaskDefinition, TaskInstance, VerifyResult

from chess_evolve.config import ELO_OPTIONS, GAMES_PER_EVAL
from chess_evolve.engine import setup_workspace
from chess_evolve.game import evaluate_pipeline
from chess_evolve.pipeline import KNOB_SPACE, PipelineConfig, build_pipeline


def _knob_values(workflow: Any) -> dict[str, Any]:
    """Extract the knob_values mapping from a candidate workflow.

    Accepts a factory Workflow (attribute access) or the raw workflow_data
    dict the outer loop persists. Returns {} when there is nothing to read.
    """
    if workflow is None:
        return {}
    values = getattr(workflow, "knob_values", None)
    if values is None and isinstance(workflow, dict):
        values = workflow.get("knob_values")
    if not isinstance(values, dict):
        return {}
    return values


def _coerce(raw: Any, default: Any) -> Any:
    """Coerce a knob value back to the type of its PipelineConfig default.

    Package.compile() widens ints to floats and bools to strings, so values
    arrive from the outer loop already stringified or float-ified.
    """
    if isinstance(default, bool):
        return str(raw).strip().lower() in {"true", "1", "yes"}
    if isinstance(default, int):
        return int(float(raw))
    if isinstance(default, float):
        return float(raw)
    return str(raw)


def config_from_workflow(opponent_elo: int, workflow: Any = None) -> PipelineConfig:
    """Build a PipelineConfig for opponent_elo, applying the candidate's knobs.

    Every knob in KNOB_SPACE that the workflow carries a value for overrides
    the PipelineConfig default. Unknown, unset or uncoercible values are left
    at their defaults, so a None workflow yields the plain seed config.
    """
    cfg = PipelineConfig(opponent_elo=opponent_elo)
    values = _knob_values(workflow)
    if not values:
        return cfg

    for knob_name, _choices in KNOB_SPACE:
        if knob_name not in values:
            continue
        default = getattr(cfg, knob_name, None)
        if default is None:
            continue
        try:
            coerced = _coerce(values[knob_name], default)
        except (TypeError, ValueError):
            continue
        if knob_name == "max_retries" and coerced < 1:
            continue  # a loop that never runs would score every game as a loss
        setattr(cfg, knob_name, coerced)

    return cfg


class ChessEvolveTask(Task):
    """Task subclass that evaluates a chess pipeline against Stockfish."""

    def __init__(self) -> None:
        defn = TaskDefinition(
            name="chess-evolve",
            description="Evaluate LLM chess play against Stockfish at various ELO levels",
            constraints=TaskConstraints(required_capabilities=[]),
        )
        super().__init__(definition=defn)

    def instances(self) -> Iterator[TaskInstance]:
        """Yield one TaskInstance per opponent ELO level."""
        for elo in ELO_OPTIONS:
            yield TaskInstance(
                id=f"elo_{elo}",
                metadata={"opponent_elo": elo, "games_per_eval": GAMES_PER_EVAL},
            )

    def setup(self, instance: TaskInstance, workspace: Path) -> None:
        """Prepare the workspace directory for evaluation."""
        setup_workspace(workspace)

    def run(
        self,
        instance: TaskInstance,
        workspace: Path,
        workflow: Any = None,
    ) -> VerifyResult:
        """Run one instance end to end: setup, then evaluate the candidate.

        Overrides Task.run, which shells out to `factory ceo --mode <workflow>`.
        Chess evaluation has no CEO phase to run — the candidate workflow is
        expressed entirely as pipeline knobs — and the outer loop's ephemeral
        mode names are not resolvable from inside a worktree, so that
        subprocess only ever fails. verify() plays the games in-process using
        the knobs carried by `workflow`.
        """
        self.setup(instance, workspace)
        return self.verify(instance, workspace, workflow)

    def verify(
        self,
        instance: TaskInstance,
        workspace: Path,
        workflow: Any = None,
    ) -> VerifyResult:
        """Run the chess evaluation pipeline and return a VerifyResult.

        Builds a pipeline with the opponent ELO from the instance metadata and
        the knob values carried by the candidate workflow (falling back to the
        default PipelineConfig when there is no workflow, so the task still
        works standalone), runs evaluate_pipeline (async), and converts the
        EvalResult into a VerifyResult whose details mirror EvalResult exactly.
        """
        opponent_elo = instance.metadata["opponent_elo"]
        n_games = instance.metadata.get("games_per_eval", GAMES_PER_EVAL)

        cfg = config_from_workflow(opponent_elo, workflow)
        pipeline = build_pipeline(cfg)

        # evaluate_pipeline is async; Task.verify is sync.
        # Use asyncio.run() to bridge the gap.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # Already inside an event loop — create a new one in a thread.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                eval_result = pool.submit(
                    asyncio.run,
                    evaluate_pipeline(pipeline, cfg, n_games=n_games),
                ).result()
        else:
            eval_result = asyncio.run(
                evaluate_pipeline(pipeline, cfg, n_games=n_games)
            )

        return VerifyResult(
            passed=(eval_result.wins + eval_result.draws) > 0,
            score=eval_result.composite_score,
            details={
                "wins": eval_result.wins,
                "draws": eval_result.draws,
                "losses": eval_result.losses,
                "total_moves": eval_result.total_moves,
                "total_errors": eval_result.total_errors,
                "total_pipeline_runs": eval_result.total_pipeline_runs,
                "games": eval_result.games,
                "score": eval_result.score,
                "avg_eval": eval_result.avg_eval,
                "blunder_count": eval_result.blunder_count,
                "composite_score": eval_result.composite_score,
                "win_rate": eval_result.win_rate,
            },
        )
