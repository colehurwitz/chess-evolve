"""Structured logging configuration for chess-evolve.

Uses ``structlog`` (available transitively via remote-factory) with a JSON
renderer for production (``CHESS_LOG_FORMAT=json``) and a coloured console
renderer for development (default).

Call ``setup_logging()`` once at process start (e.g. in ``cli.main()``).
"""

from __future__ import annotations

import os
import sys

import structlog


def setup_logging() -> None:
    """Configure structlog processors and renderer.

    Renderer is chosen by the ``CHESS_LOG_FORMAT`` environment variable:
    - ``json``    → machine-readable JSON lines on stderr
    - ``console`` → human-readable coloured output (default)
    """
    fmt = os.environ.get("CHESS_LOG_FORMAT", "console").lower()

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if fmt == "json":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(0),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
