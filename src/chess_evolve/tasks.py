"""PositionTask — per-position move-quality evaluation as a factory Task.

Implements the real four-hook ``factory.task.Task`` interface
(``instances`` / ``setup`` / ``prompt`` / ``verify``) so a ``DataNode`` with
``task_ref='chess_evolve.tasks:PositionTask'`` can drive the move-generator
subgraph once per position and score each chosen move against Stockfish's best
move (centipawn loss).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

import chess
import chess.engine
from factory.task import Task, TaskInstance, VerifyResult

from chess_evolve.config import resolve_stockfish
from chess_evolve.engine import read_move, write_board_state

# Scores at/above this magnitude are treated as "mate-scale" (checkmate found).
_MATE_SCALE = 9000
# Default analysis depth when no explicit depth/time is configured.
_DEFAULT_DEPTH = 12


def _default_positions_file() -> Path:
    """Resolve the positions JSON path (env override -> repo default)."""
    env = os.environ.get("CHESS_POSITIONS_FILE")
    if env:
        return Path(env)
    return Path("eval/test_positions.json")


def load_positions(positions_file: str | Path | None = None) -> list[dict]:
    """Load and validate positions from an EPD-inspired JSON file.

    Each entry must contain a valid ``fen``; invalid FENs fail fast with a
    clear error (no silent skipping).
    """
    path = Path(positions_file) if positions_file else _default_positions_file()
    if not path.exists():
        raise FileNotFoundError(f"Positions file not found: {path}")
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(
            f"Positions file {path} must contain a JSON array of positions."
        )
    positions: list[dict] = []
    for idx, entry in enumerate(raw):
        fen = entry.get("fen")
        if not fen:
            raise ValueError(f"Position #{idx} in {path} is missing a 'fen' field.")
        try:
            chess.Board(fen)  # validate defensively
        except ValueError as exc:
            raise ValueError(
                f"Position #{idx} ({entry.get('id', '?')}) has invalid FEN "
                f"{fen!r}: {exc}"
            ) from exc
        positions.append({
            "id": entry.get("id", f"pos-{idx}"),
            "fen": fen,
            "phase": entry.get("phase", ""),
            "description": entry.get("description", ""),
        })
    if not positions:
        raise ValueError(f"Positions file {path} contains no positions.")
    return positions


def _analysis_limit() -> chess.engine.Limit:
    """Build the Stockfish analysis limit from env config (time preferred)."""
    time_s = os.environ.get("CHESS_EVAL_TIME")
    if time_s:
        return chess.engine.Limit(time=float(time_s))
    depth = os.environ.get("CHESS_EVAL_DEPTH")
    return chess.engine.Limit(depth=int(depth) if depth else _DEFAULT_DEPTH)


class PositionTask(Task):
    """Evaluate a single chess position: generate a move, score its CPL."""

    def instances(self) -> Iterator[TaskInstance]:
        """Yield one TaskInstance per position, carrying the FEN in metadata."""
        for pos in load_positions():
            yield TaskInstance(
                id=pos["id"],
                metadata={
                    "fen": pos["fen"],
                    "phase": pos["phase"],
                    "description": pos["description"],
                },
            )

    def setup(self, instance: TaskInstance, workspace: Path) -> None:
        """Write the position's board state into the workspace.

        Does NOT call ``setup_workspace`` (which rmtrees the tree); only creates
        the ``.factory/chess`` directory and writes ``board_state.md``.
        """
        workspace = Path(workspace)
        chess_dir = workspace / ".factory" / "chess"
        chess_dir.mkdir(parents=True, exist_ok=True)
        board = chess.Board(instance.metadata["fen"])
        write_board_state(workspace, board)
        (chess_dir / "memory.md").write_text("")

    def prompt(self, instance: TaskInstance) -> str:
        """Return the board prompt text for this position."""
        board = chess.Board(instance.metadata["fen"])
        legal_moves = [m.uci() for m in board.legal_moves]
        return (
            f"Position (FEN): {board.fen()}\n"
            f"You are playing {'White' if board.turn else 'Black'}.\n"
            f"Legal moves: {', '.join(legal_moves)}\n\n"
            f"Board:\n{board}\n\n"
            "Pick the best move. Output ONLY the UCI move (e.g. e2e4)."
        )

    def verify(self, instance: TaskInstance, workspace: Path) -> VerifyResult:
        """Score the chosen move by centipawn loss vs Stockfish's best move."""
        workspace = Path(workspace)
        fen = instance.metadata["fen"]
        board = chess.Board(fen)

        # Terminal positions: no move to make — nothing to score.
        if board.is_game_over():
            return VerifyResult(
                passed=True,
                score=1.0,
                details={"note": "terminal position, no move required", "fen": fen},
            )

        chosen = read_move(workspace, board)

        stockfish = resolve_stockfish()
        if stockfish is None:
            return VerifyResult(
                passed=False,
                score=0.0,
                details={
                    "error": "stockfish_not_found",
                    "move": chosen,
                    "fen": fen,
                },
            )

        limit = _analysis_limit()
        engine = chess.engine.SimpleEngine.popen_uci(stockfish)
        try:
            best_info = engine.analyse(board, limit)
            best_move = best_info.get("pv", [None])[0]
            best_move_uci = best_move.uci() if best_move else None
            best_value = best_info["score"].pov(board.turn).score(
                mate_score=_MATE_SCALE + 1000,
            )

            if chosen is None:
                # No legal move parsed from the pipeline output — worst case.
                cpl = float(best_value if best_value > 0 else 0) + 1000.0
                chosen_value = None
            else:
                after = board.copy()
                after.push_uci(chosen)
                if after.is_game_over():
                    # Checkmate/stalemate reached by the chosen move.
                    if after.is_checkmate():
                        chosen_value = _MATE_SCALE + 1000
                    else:
                        chosen_value = 0
                else:
                    after_info = engine.analyse(after, limit)
                    # Score is relative to the side to move in `after` (opponent),
                    # so negate to get the value for the original mover.
                    chosen_value = -after_info["score"].pov(after.turn).score(
                        mate_score=_MATE_SCALE + 1000,
                    )
                # Both moves are winning mates: treat as no loss.
                if best_value >= _MATE_SCALE and chosen_value >= _MATE_SCALE:
                    cpl = 0.0
                else:
                    cpl = float(max(0, best_value - chosen_value))
        finally:
            engine.quit()

        score = max(0.0, 1.0 - cpl / 100.0)
        passed = chosen is not None and cpl < 25
        return VerifyResult(
            passed=passed,
            score=score,
            details={
                "cpl": cpl,
                "move": chosen,
                "best_move": best_move_uci,
                "fen": fen,
                "phase": instance.metadata.get("phase", ""),
            },
        )
