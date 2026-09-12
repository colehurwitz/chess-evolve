"""Broadcast module — stub for game state broadcasting.

The original broadcast implementation was removed during the dead-code
cleanup.  ``engine.py`` still imports ``broadcast_game_state`` — this
stub keeps the import valid while the feature is not wired.
"""

from __future__ import annotations

from typing import Any


def broadcast_game_state(*args: Any, **kwargs: Any) -> None:
    """Broadcast game state to listeners (no-op stub)."""
