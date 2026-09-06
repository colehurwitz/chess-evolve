"""Factory Task abstraction for chess-evolve.

Wraps the existing evaluation harness (evaluate_pipeline, build_pipeline,
setup_workspace) behind factory's four-hook Task interface so the factory
outer loop can drive chess evaluation via Task.instances / setup / prompt /
verify.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Iterator

from factory.task import Task, TaskDefinition, TaskInstance, VerifyResult

from chess_evolve.config import ELO_OPTIONS, GAMES_PER_EVAL, MAX_MOVES
from chess_evolve.engine import setup_workspace
from chess_evolve.game import evaluate_pipeline
from chess_evolve.pipeline import PipelineConfig, build_pipeline

PROMPT_TEMPLATE = """\
# Improve the chess pipeline against Stockfish (ELO {opponent_elo})

The pipeline in `src/chess_evolve/` plays chess by prompting an LLM for one
move at a time. It currently loses. Make it play stronger chess against a
Stockfish opponent rated {opponent_elo} ELO.

## How a move is produced

For every LLM turn the harness writes `.factory/chess/board_state.md` into the
workspace containing the FEN, which colour you play, the full list of legal
UCI moves, and an ASCII board. The `generator` agent reads that file and must
write a legal UCI move (e.g. `e2e4`) to `.factory/chess/move.md`. A legality
gate then re-runs the generator up to `max_retries` times if the move it wrote
is not in the legal-move list.

## What you may change

- `src/chess_evolve/prompts.py` — the prompt registry driving the generator.
- `src/chess_evolve/pipeline.py` — `KNOB_SPACE`, `PipelineConfig`, and the
  `build_pipeline()` topology (you may add nodes, gates, and loops).

Keep the output contract intact: whatever the generator writes, a legal UCI
move must be recoverable from `.factory/chess/move.md`. Prompts that explain
chess strategy help; prompts that break the output format score zero.

## How you are scored

{n_games} game(s) are played per evaluation, capped at {max_moves} moves each.
Every position is scored by Stockfish in centipawns from your side's point of
view, and `composite_score` sums `(cp + 500) / 1000` over the evaluation curve
for as long as you stay above -500cp — so surviving longer without a losing
position is worth more than a short sharp game. A win adds 200, a draw 100.
The instance passes only if at least one game is a win or a draw.

Optimise for `composite_score`: avoid blunders, avoid illegal moves (each one
burns a retry), and keep the position playable.
"""


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

    def prompt(self, instance: TaskInstance) -> str:
        """Describe the chess optimisation goal for one instance.

        Overrides the generic base-class prompt so the agent driven by
        Task.run() is told which opponent ELO it faces, how moves are
        produced, which surfaces it may change, and how composite_score
        is computed.
        """
        opponent_elo = instance.metadata.get("opponent_elo", PipelineConfig().opponent_elo)
        n_games = instance.metadata.get("games_per_eval", GAMES_PER_EVAL)

        return PROMPT_TEMPLATE.format(
            opponent_elo=opponent_elo,
            n_games=n_games,
            max_moves=MAX_MOVES,
        )

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
