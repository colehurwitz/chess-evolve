"""Live UI broadcasting — no-op stub.

The original module was deleted during dead-code cleanup but ``engine.py``
still imports ``broadcast_game_state``.  This stub satisfies the import
and silently drops every call so the rest of the system keeps working.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import chess


def broadcast_game_state(
    tag: str,
    board: chess.Board,
    game_moves: list[str],
    llm_white: bool,
    elo: int,
    result: str | None = None,
    move_count: int = 0,
    gen: int = 0,
    config: str = "",
    whose_turn: str = "",
    active_node: str = "",
    node_outputs: dict | None = None,
    eval_curve: list[int] | None = None,
    full_config: str = "",
) -> None:
    """No-op: accepts game state but does nothing."""
