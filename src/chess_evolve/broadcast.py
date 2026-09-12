"""Broadcast helpers for live game-state updates to the dashboard.

Provides ``broadcast_game_state`` which publishes board snapshots to the
SSE event stream consumed by the dashboard frontend.
"""

from __future__ import annotations

from typing import Any

import chess


def broadcast_game_state(
    game_tag: str,
    board: chess.Board,
    game_moves: list[str],
    llm_white: bool,
    stockfish_elo: int,
    *,
    move_count: int = 0,
    gen: int = 0,
    config: str = "",
    whose_turn: str = "llm",
    active_node: str = "",
    node_outputs: dict[str, Any] | None = None,
    eval_curve: list[int] | None = None,
    full_config: str = "",
) -> None:
    """Broadcast current game state to dashboard listeners.

    This is a best-effort operation — if no dashboard is running or the
    broadcast fails, the game continues unaffected.
    """
    # Currently a no-op stub.  The dashboard reads game state directly
    # from the workspace files written by advance_game_state().
    pass
