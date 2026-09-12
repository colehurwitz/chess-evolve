#!/usr/bin/env python3
"""Standalone game gate script for chess-game workflow.

Called by GateNode evaluator_command. Advances the game state by one ply-pair
(LLM move + Stockfish response). Prints PROCEED (game over) or RELOOP (continue).

Self-contained: imports only from chess_evolve.tasks and chess_evolve.config
(not chess_evolve.engine, which has a broken broadcast import).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import chess
import chess.engine

from chess_evolve.config import MAX_MOVES, resolve_stockfish
from chess_evolve.tasks import read_move, write_board_state

# ── Helpers (copied from engine.py lines 530-598) ──────────────


def _play_stockfish_move(
    board: chess.Board, state: dict, stockfish_path: str,
) -> None:
    """Play one Stockfish move, append to state, increment move_count."""
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    try:
        engine.configure({
            "UCI_LimitStrength": True,
            "UCI_Elo": state["opponent_elo"],
        })
        result = engine.play(board, chess.engine.Limit(time=0.1))
        move_uci = result.move.uci()
        board.push(result.move)
        state["move_list"].append(move_uci)
        state["move_count"] += 1
    finally:
        engine.quit()


def _eval_and_append(
    board: chess.Board,
    state: dict,
    llm_plays_white: bool,
    stockfish_path: str,
) -> None:
    """Analyse position with Stockfish and append centipawn eval to state."""
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    try:
        info = engine.analyse(board, chess.engine.Limit(time=0.05))
        cp = info["score"].white().score(mate_score=10000)
        eval_cp = cp if llm_plays_white else -cp
        state["eval_curve"].append(eval_cp)
    finally:
        engine.quit()


def _check_game_over(
    board: chess.Board,
    state: dict,
    llm_plays_white: bool,
    chess_dir: Path,
) -> bool:
    """Check if game is over, set result, write state. Return True if done."""
    is_over = board.is_game_over() or state["move_count"] >= MAX_MOVES
    if not is_over:
        return False
    outcome = board.outcome()
    if outcome is not None and outcome.winner is not None:
        if (outcome.winner == chess.WHITE) == llm_plays_white:
            state["result"] = "win"
        else:
            state["result"] = "loss"
    elif outcome is not None and outcome.winner is None:
        state["result"] = "draw"
    else:
        # Max moves or unclear — use final eval
        curve = state["eval_curve"]
        final = curve[-1] if curve else 0
        if final > 100:
            state["result"] = "draw"
        elif final < -100:
            state["result"] = "loss"
        else:
            state["result"] = "draw"
    state["game_over"] = True
    state["fen"] = board.fen()
    (chess_dir / "game_state.json").write_text(json.dumps(state))
    return True


# ── Main gate logic (copied from engine.py lines 600-680) ──────


def advance_game_state(project_path: str) -> None:
    """Advance game by one ply-pair: apply LLM move, play Stockfish, update.

    Called by ``game_gate`` GateNode evaluator_command.
    Prints ``PROCEED`` (game over) or ``RELOOP`` (continue).
    """
    workspace = Path(project_path)
    chess_dir = workspace / ".factory" / "chess"
    state = json.loads((chess_dir / "game_state.json").read_text())

    stockfish_path = resolve_stockfish()
    if stockfish_path is None:
        # No Stockfish — can't play. Mark as loss and exit.
        state["result"] = "loss"
        state["game_over"] = True
        (chess_dir / "game_state.json").write_text(json.dumps(state))
        print("PROCEED")
        return

    board = chess.Board(state["fen"])
    llm_plays_white = state["color"] == "white"
    is_llm_turn = (board.turn == chess.WHITE) == llm_plays_white

    # If it's NOT the LLM's turn (Stockfish moves first when LLM plays black
    # on move 1), play Stockfish move first
    if not is_llm_turn:
        _play_stockfish_move(board, state, stockfish_path)
        if _check_game_over(board, state, llm_plays_white, chess_dir):
            print("PROCEED")
            return
        # Now it's LLM's turn — update board_state.md for generator
        state["fen"] = board.fen()
        write_board_state(workspace, board)
        (chess_dir / "game_state.json").write_text(json.dumps(state))
        print("RELOOP")
        return

    # LLM's turn: read the move from move.md (written by generator via
    # node.writes)
    chosen = read_move(workspace, board)
    if chosen is None:
        state["illegal_attempts"] += 1
        chosen = random.choice(list(board.legal_moves)).uci()

    board.push_uci(chosen)
    state["move_list"].append(chosen)
    state["move_count"] += 1

    # Evaluate position after LLM's move
    _eval_and_append(board, state, llm_plays_white, stockfish_path)

    if _check_game_over(board, state, llm_plays_white, chess_dir):
        print("PROCEED")
        return

    # Play Stockfish's response
    _play_stockfish_move(board, state, stockfish_path)

    if _check_game_over(board, state, llm_plays_white, chess_dir):
        print("PROCEED")
        return

    # Evaluate position after Stockfish's move
    _eval_and_append(board, state, llm_plays_white, stockfish_path)

    # Early resignation: eval < -500cp for 2 consecutive moves
    curve = state["eval_curve"]
    if len(curve) >= 2 and all(e < -500 for e in curve[-2:]):
        state["result"] = "loss"
        state["game_over"] = True
        state["fen"] = board.fen()
        (chess_dir / "game_state.json").write_text(json.dumps(state))
        print("PROCEED")
        return

    # Update board state for next generator iteration
    state["fen"] = board.fen()
    (chess_dir / "game_state.json").write_text(json.dumps(state))
    write_board_state(workspace, board)
    print("RELOOP")


if __name__ == "__main__":
    advance_game_state(sys.argv[1])
