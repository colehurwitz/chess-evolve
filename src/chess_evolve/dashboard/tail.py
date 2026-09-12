"""JSONL file tailer with seek-based incremental reading."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncIterator


def parse_event(line: str) -> dict | None:
    """Parse a single event line into a dict.

    Currently handles JSONL. Designed to be extended later for
    structured YAML frontmatter or other formats.
    """
    stripped = line.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return None


async def tail_jsonl(
    path: Path,
    *,
    poll_interval: float = 0.5,
    from_beginning: bool = False,
    replay_delay: float = 0.1,
) -> AsyncIterator[dict]:
    """Yield new JSON events from a JSONL file using seek-based reads.

    Args:
        path: Path to the JSONL file.
        poll_interval: Seconds between poll cycles (fallback).
        from_beginning: If True, read from start (replay mode).
        replay_delay: Delay between events in replay mode.
    """
    position = 0
    remainder = ""

    # Optional watchfiles integration for lower-latency notifications
    _watch_changes = None
    try:
        from watchfiles import awatch  # type: ignore[import-untyped]

        async def _watcher(p: Path) -> AsyncIterator[None]:
            async for _ in awatch(p.parent):
                yield

        _watch_changes = _watcher
    except ImportError:
        pass

    while True:
        try:
            with open(path, "rb") as fh:
                fh.seek(position)
                raw = fh.read()
        except FileNotFoundError:
            await asyncio.sleep(poll_interval)
            continue

        if not raw:
            if from_beginning and position > 0:
                # Replay mode: we've read everything, stop
                return
            if _watch_changes is not None:
                try:
                    async for _ in _watch_changes(path):
                        break
                except Exception:
                    await asyncio.sleep(poll_interval)
            else:
                await asyncio.sleep(poll_interval)
            continue

        text = remainder + raw.decode("utf-8", errors="replace")
        lines = text.split("\n")

        # Keep the last element as remainder (may be incomplete)
        remainder = lines[-1]
        complete_lines = lines[:-1]

        position += len(raw)

        for line in complete_lines:
            event = parse_event(line)
            if event is not None:
                yield event
                if from_beginning:
                    await asyncio.sleep(replay_delay)

        if from_beginning and not complete_lines and not raw:
            return
