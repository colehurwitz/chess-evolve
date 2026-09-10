"""Tests for the live evolution dashboard."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from chess_evolve.dashboard import create_app


@pytest_asyncio.fixture
async def project_dir(tmp_path: Path) -> Path:
    """Create a minimal project directory with factory artifacts."""
    factory = tmp_path / ".factory"
    factory.mkdir()
    events = factory / "events.jsonl"
    ev1 = {"type": "experiment.started", "timestamp": "2026-01-01T00:00:00Z",
           "data": {"score": 0.5}}
    ev2 = {"type": "experiment.completed", "timestamp": "2026-01-01T00:01:00Z",
           "data": {"score": 0.8, "cost": 0.05, "diversity": 0.42}}
    events.write_text(json.dumps(ev1) + "\n" + json.dumps(ev2) + "\n")
    config = factory / "config.json"
    config.write_text(json.dumps({"project": "test", "version": 1}))
    return tmp_path


@pytest_asyncio.fixture
async def client(project_dir: Path) -> httpx.AsyncClient:
    app = create_app(project_root=project_dir)
    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c  # type: ignore[misc]


@pytest.mark.asyncio
async def test_health_endpoint(client: httpx.AsyncClient) -> None:
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_root_returns_html(client: httpx.AsyncClient) -> None:
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Dashboard" in resp.text


@pytest.mark.asyncio
async def test_events_sse_content_type(project_dir: Path) -> None:
    # Use replay mode to get a finite stream (avoids hanging on infinite SSE)
    events_file = project_dir / ".factory" / "events.jsonl"
    app = create_app(project_root=project_dir, replay_path=events_file, replay_delay=0.01)
    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        async with c.stream("GET", "/events") as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]


@pytest.mark.asyncio
async def test_events_streams_prepopulated_data(project_dir: Path) -> None:
    # Use replay mode so the stream terminates after reading all events
    events_file = project_dir / ".factory" / "events.jsonl"
    app = create_app(project_root=project_dir, replay_path=events_file, replay_delay=0.01)
    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
    collected: list[dict] = []

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        try:
            async with asyncio.timeout(5.0):
                async with c.stream("GET", "/events") as resp:
                    async for line in resp.aiter_lines():
                        if line.startswith("data:"):
                            data_str = line[len("data:"):].strip()
                            collected.append(json.loads(data_str))
        except TimeoutError:
            pass

    assert len(collected) >= 1
    assert collected[0]["type"] == "experiment.started"


@pytest.mark.asyncio
async def test_events_handles_missing_file(tmp_path: Path) -> None:
    """Dashboard should not crash when events.jsonl doesn't exist yet."""
    app = create_app(project_root=tmp_path)
    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        try:
            async with asyncio.timeout(1.5):
                async with c.stream("GET", "/events") as resp:
                    assert resp.status_code == 200
                    # Should not crash — just waits for file
                    async for line in resp.aiter_lines():
                        break  # If we get anything, that's fine
        except TimeoutError:
            pass  # Expected — no file means no events, timeout is OK


@pytest.mark.asyncio
async def test_events_handles_malformed_json(tmp_path: Path) -> None:
    factory = tmp_path / ".factory"
    factory.mkdir()
    events = factory / "events.jsonl"
    events.write_text(
        "not valid json\n"
        + json.dumps({"type": "valid.event", "data": {}}) + "\n"
    )
    # Use replay mode so the stream terminates after reading all lines
    app = create_app(project_root=tmp_path, replay_path=events, replay_delay=0.01)
    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
    collected: list[dict] = []

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        try:
            async with asyncio.timeout(5.0):
                async with c.stream("GET", "/events") as resp:
                    async for line in resp.aiter_lines():
                        if line.startswith("data:"):
                            data_str = line[len("data:"):].strip()
                            collected.append(json.loads(data_str))
        except TimeoutError:
            pass

    # The malformed line is skipped, the valid event is streamed
    assert len(collected) == 1
    assert collected[0]["type"] == "valid.event"


@pytest.mark.asyncio
async def test_state_endpoint(client: httpx.AsyncClient) -> None:
    resp = await client.get("/state")
    assert resp.status_code == 200
    body = resp.json()
    assert "generation" in body
    assert "latest_events" in body
    assert "config" in body
    assert body["config"]["project"] == "test"
    assert body["generation"] >= 1  # experiment.completed counts


@pytest.mark.asyncio
async def test_replay_mode(tmp_path: Path) -> None:
    replay_file = tmp_path / "replay.jsonl"
    events = [
        {"type": "gen.1", "data": {"score": 0.3}},
        {"type": "gen.2", "data": {"score": 0.6}},
        {"type": "gen.3", "data": {"score": 0.9}},
    ]
    replay_file.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    app = create_app(project_root=tmp_path, replay_path=replay_file, replay_delay=0.01)
    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
    collected: list[dict] = []

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        try:
            async with asyncio.timeout(5.0):
                async with c.stream("GET", "/events") as resp:
                    async for line in resp.aiter_lines():
                        if line.startswith("data:"):
                            data_str = line[len("data:"):].strip()
                            collected.append(json.loads(data_str))
        except TimeoutError:
            pass

    assert len(collected) == 3
    assert collected[0]["type"] == "gen.1"
    assert collected[2]["type"] == "gen.3"
