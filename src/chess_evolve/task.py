"""Factory Task abstraction for chess-evolve.

Wraps the existing evaluation harness (evaluate_pipeline, build_pipeline,
setup_workspace) behind factory's four-hook Task interface so the factory
outer loop can drive chess evaluation via Task.instances / setup / verify.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, ConfigDict

from chess_evolve.config import ELO_OPTIONS, GAMES_PER_EVAL
from chess_evolve.engine import setup_workspace
from chess_evolve.game import evaluate_pipeline
from chess_evolve.pipeline import PipelineConfig, build_pipeline


class VerifyResult(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    passed: bool
    score: float
    details: dict[str, object] = {}


@dataclass
class TaskDefinition:
    name: str
    description: str


@dataclass
class TaskInstance:
    id: str
    metadata: dict[str, object] = field(default_factory=dict)


class Task:
    """Minimal base class for factory Task interface."""

    def __init__(self, definition: TaskDefinition) -> None:
        self._definition = definition

    @property
    def name(self) -> str:
        return self._definition.name


class ChessEvolveTask(Task):
    """Task subclass that evaluates a chess pipeline against Stockfish."""

    def __init__(self) -> None:
        defn = TaskDefinition(
            name="chess-evolve",
            description="Evaluate LLM chess play against Stockfish at various ELO levels",
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

    def verify(self, instance: TaskInstance, workspace: Path) -> VerifyResult:
        """Run the chess evaluation pipeline and return a VerifyResult.

        Builds a pipeline with the opponent ELO from the instance metadata,
        runs evaluate_pipeline (async), and converts the EvalResult into a
        VerifyResult whose details mirror EvalResult exactly.
        """
        opponent_elo = instance.metadata["opponent_elo"]
        n_games = instance.metadata.get("games_per_eval", GAMES_PER_EVAL)

        cfg = PipelineConfig(opponent_elo=opponent_elo)
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
