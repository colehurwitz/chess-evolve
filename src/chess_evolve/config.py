"""Constants and configuration for chess-evolve."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

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


# Best-effort resolved path; falls back to the bare command name so that
# environments with Stockfish on PATH still work via SimpleEngine.popen_uci.
STOCKFISH_PATH = resolve_stockfish() or "stockfish"
ELO_OPTIONS = [1320, 1420, 1520, 1620]
GAMES_PER_EVAL = 1
MAX_MOVES = 60
NUM_GENERATIONS = 6
CANDIDATES_PER_GEN = 5
WORKSPACE = Path(os.environ.get("CHESS_WORKSPACE", "/tmp/chess-factory"))
LIVE_DIR = WORKSPACE / "live"
