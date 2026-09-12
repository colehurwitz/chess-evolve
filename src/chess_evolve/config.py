"""Constants and configuration for chess-evolve."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

# ELO levels used by GameTask to generate opponent instances.
ELO_OPTIONS: tuple[int, ...] = (1320,)

# Maximum half-moves (plies) per game before forced termination.
MAX_MOVES: int = 80

# Common install locations checked as a last resort (no hardcoded single path).
_STOCKFISH_COMMON_PATHS = (
    "/usr/bin/stockfish",
    "/usr/local/bin/stockfish",
    "/usr/games/stockfish",
    "/opt/homebrew/bin/stockfish",
    "/home/linuxbrew/.linuxbrew/bin/stockfish",
)


def resolve_stockfish() -> str | None:
    """Resolve the Stockfish binary portably.

    Resolution order: ``STOCKFISH_PATH`` env var -> ``shutil.which('stockfish')``
    -> a small set of common install paths. Returns ``None`` if not found so
    callers can degrade gracefully instead of crashing.
    """
    env = os.environ.get("STOCKFISH_PATH")
    if env and Path(env).exists():
        return env
    found = shutil.which("stockfish")
    if found:
        return found
    for candidate in _STOCKFISH_COMMON_PATHS:
        if Path(candidate).exists():
            return candidate
    return None
