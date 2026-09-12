"""PositionTask & GameTask — factory Task implementations for chess-evolve.

PositionTask: per-position move-quality evaluation (centipawn loss).
GameTask: full-game evaluation with rich verify() details for reflector.

Both implement the four-hook ``factory.task.Task`` interface
(``instances`` / ``setup`` / ``prompt`` / ``verify``) so a ``DataNode`` can
drive a subgraph and score the result.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterator

import chess
import chess.engine
from factory.task import Task, TaskInstance, VerifyResult

from chess_evolve.config import ELO_OPTIONS, resolve_stockfish


def write_board_state(workspace: Path, board: chess.Board) -> None:
    """Write the current board state as an artifact for the pipeline to read."""
    legal_moves = [move.uci() for move in board.legal_moves]
    content = (
        f"FEN: {board.fen()}\n\n"
        f"You are playing {'White' if board.turn else 'Black'}.\n\n"
        f"Legal moves: {', '.join(legal_moves)}\n\n"
        f"Board:\n{board}\n"
    )
    (workspace / ".factory" / "chess" / "board_state.md").write_text(content)


def read_move(workspace: Path, board: chess.Board) -> str | None:
    """Read the move chosen by the pipeline from the artifact file."""
    move_file = workspace / ".factory" / "chess" / "move.md"
    if not move_file.exists():
        return None
    content = move_file.read_text().strip()
    legal_moves = [m.uci() for m in board.legal_moves]
    for token in content.split():
        cleaned = token.strip(".,!()[]{}\"'`\n")
        if cleaned in legal_moves:
            return cleaned
    for match in re.finditer(r'[a-h][1-8][a-h][1-8][qrbn]?', content):
        if match.group() in legal_moves:
            return match.group()
    return None

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


# ── GameTask — full game evaluation ─────────────────────────────


def _extract_blunders(
    eval_curve: list[int], move_list: list[str],
) -> list[dict]:
    """Extract structured blunder records from an eval curve.

    A blunder is a >200 centipawn drop between consecutive evaluations.
    Returns a list of dicts with move context for the reflector.
    """
    blunders: list[dict] = []
    for i in range(1, len(eval_curve)):
        drop = eval_curve[i] - eval_curve[i - 1]
        if drop < -200:
            blunders.append({
                "move_index": i,
                "move_num": i // 2 + 1,
                "move_uci": move_list[i] if i < len(move_list) else "?",
                "cp_before": eval_curve[i - 1],
                "cp_after": eval_curve[i],
                "cp_drop": drop,
            })
    return blunders


class GameTask(Task):
    """Evaluate a full game against Stockfish, returning rich details.

    ``verify()`` is a **scorer only** — it reads ``game_state.json`` from the
    workspace (written by the ``game_gate`` GateNode calling
    ``advance_game_state``) and scores the completed game.  The game loop
    itself lives in the workflow subgraph (``Loop(generator → game_gate)``).
    """

    def instances(self) -> Iterator[TaskInstance]:
        """Yield one ``TaskInstance`` per ELO × color combination."""
        for elo in ELO_OPTIONS:
            for color in ("white", "black"):
                yield TaskInstance(
                    id=f"elo{elo}_{color}",
                    metadata={"opponent_elo": elo, "color": color},
                )

    def setup(self, instance: TaskInstance, workspace: Path) -> None:
        """Write starting position and initialize ``game_state.json``."""
        workspace = Path(workspace)
        chess_dir = workspace / ".factory" / "chess"
        chess_dir.mkdir(parents=True, exist_ok=True)
        board = chess.Board()
        write_board_state(workspace, board)
        game_state = {
            "opponent_elo": instance.metadata["opponent_elo"],
            "color": instance.metadata["color"],
            "fen": board.fen(),
            "move_list": [],
            "eval_curve": [],
            "illegal_attempts": 0,
            "move_count": 0,
            "result": None,
            "game_over": False,
        }
        (chess_dir / "game_state.json").write_text(json.dumps(game_state))

    def prompt(self, instance: TaskInstance) -> str:
        """Return the starting board prompt for this game."""
        board = chess.Board()
        legal_moves = [m.uci() for m in board.legal_moves]
        color = instance.metadata["color"]
        return (
            f"Position (FEN): {board.fen()}\n"
            f"You are playing {'White' if color == 'white' else 'Black'}.\n"
            f"Legal moves: {', '.join(legal_moves)}\n\n"
            f"Board:\n{board}\n\n"
            "Pick the best move. Output ONLY the UCI move (e.g. e2e4)."
        )

    def verify(self, instance: TaskInstance, workspace: Path) -> VerifyResult:
        """Score a completed game from ``game_state.json``.

        Does NOT call ``evaluate_pipeline()``, ``play_game()``, or any
        game-loop function.  Reads the workspace file only.
        """
        workspace = Path(workspace)
        state_path = workspace / ".factory" / "chess" / "game_state.json"
        if not state_path.exists():
            return VerifyResult(
                passed=False,
                score=0.0,
                details={"error": "game_state_not_found"},
            )
        state = json.loads(state_path.read_text())
        result = state.get("result", "loss")
        eval_curve: list[int] = state.get("eval_curve", [])
        move_list: list[str] = state.get("move_list", [])
        color: str = state.get("color", "white")
        elo: int = state.get("opponent_elo", 0)
        illegal: int = state.get("illegal_attempts", 0)

        # Extract structured blunders from eval curve (>200cp drops)
        blunders = _extract_blunders(eval_curve, move_list)
        avg_cp = (
            sum(eval_curve) / max(len(eval_curve), 1) if eval_curve else 0.0
        )

        # Composite score: reuse EvalResult formula
        threshold = -500
        position_score = sum(
            (cp - threshold) / 1000.0
            for cp in eval_curve
            if cp >= threshold
        )
        outcome_bonus = {"win": 200, "draw": 100}.get(result, 0)
        composite = position_score + outcome_bonus

        passed = result in ("win", "draw")
        return VerifyResult(
            passed=passed,
            score=composite,
            details={
                "result": result,
                "move_list": move_list,
                "eval_curve": eval_curve,
                "blunders": blunders,
                "illegal_attempts": illegal,
                "avg_centipawn": avg_cp,
                "opponent_elo": elo,
                "color": color,
                "move_count": len(move_list),
                "composite_score": composite,
            },
        )
