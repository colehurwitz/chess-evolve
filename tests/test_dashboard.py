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


# ---------------------------------------------------------------------------
# New tests for /generations, /games, /outer-loop/events endpoints
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generations_endpoint(tmp_path: Path) -> None:
    """Mock reflections files and verify /generations returns sorted array."""
    ref_dir = tmp_path / ".factory" / "outer_loop" / "reflections"
    ref_dir.mkdir(parents=True)

    gen0 = {
        "generation": 0,
        "typed_suggestions": [
            {"operator": "prompt_mutate", "target": "gen", "rationale": "r0"},
            {"operator": "knob_mutate", "target": "lr", "rationale": "r1"},
        ],
        "top_k_ids": ["aaa"],
        "bottom_k_ids": ["bbb"],
        "failure_patterns": ["fail0"],
        "success_patterns": ["ok0"],
    }
    gen1 = {
        "generation": 1,
        "typed_suggestions": [
            {"operator": "node_insert", "target": "val", "rationale": "r2"},
        ],
        "top_k_ids": ["ccc"],
        "bottom_k_ids": ["ddd"],
        "failure_patterns": [],
        "success_patterns": ["ok1"],
    }
    # Write gen1 first to verify sorting
    (ref_dir / "gen1.json").write_text(json.dumps(gen1))
    (ref_dir / "gen0.json").write_text(json.dumps(gen0))

    app = create_app(project_root=tmp_path)
    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/generations")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert len(body) == 2
    assert body[0]["generation"] == 0
    assert body[1]["generation"] == 1
    assert len(body[0]["typed_suggestions"]) == 2
    assert body[0]["typed_suggestions"][0]["operator"] == "prompt_mutate"
    assert body[1]["top_k_ids"] == ["ccc"]


@pytest.mark.asyncio
async def test_generations_empty(tmp_path: Path) -> None:
    """No reflections directory → returns []."""
    app = create_app(project_root=tmp_path)
    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/generations")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_games_endpoint(tmp_path: Path) -> None:
    """Mock eval worktree with game_state.json, verify /games response."""
    wt_dir = tmp_path / "eval-wts" / "wt-abc123"
    chess_dir = wt_dir / ".factory" / "chess"
    chess_dir.mkdir(parents=True)

    game = {
        "opponent_elo": 1320,
        "color": "white",
        "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "move_list": ["e2e4"],
        "eval_curve": [30],
        "illegal_attempts": 0,
        "move_count": 1,
        "result": None,
        "game_over": False,
    }
    (chess_dir / "game_state.json").write_text(json.dumps(game))

    app = create_app(project_root=tmp_path, eval_worktrees_dir=tmp_path / "eval-wts")
    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/games")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert len(body) == 1
    assert body[0]["worktree_id"] == "wt-abc123"
    assert body[0]["opponent_elo"] == 1320
    assert body[0]["fen"].startswith("rnbqkbnr")
    assert body[0]["move_list"] == ["e2e4"]


@pytest.mark.asyncio
async def test_games_no_worktrees(tmp_path: Path) -> None:
    """Missing eval dir → returns []."""
    app = create_app(
        project_root=tmp_path,
        eval_worktrees_dir=tmp_path / "nonexistent",
    )
    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/games")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_games_corrupted_json(tmp_path: Path) -> None:
    """Corrupted game_state.json → skipped, no crash."""
    wt_dir = tmp_path / "eval-wts" / "wt-bad"
    chess_dir = wt_dir / ".factory" / "chess"
    chess_dir.mkdir(parents=True)
    (chess_dir / "game_state.json").write_text("{corrupt json!!!")

    # Also add a valid worktree to verify it's still returned
    wt_good = tmp_path / "eval-wts" / "wt-good"
    chess_good = wt_good / ".factory" / "chess"
    chess_good.mkdir(parents=True)
    (chess_good / "game_state.json").write_text(
        json.dumps({"fen": "startpos", "game_over": False})
    )

    app = create_app(
        project_root=tmp_path,
        eval_worktrees_dir=tmp_path / "eval-wts",
    )
    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/games")
    assert resp.status_code == 200
    body = resp.json()
    # Corrupted one skipped, valid one returned
    assert len(body) == 1
    assert body[0]["worktree_id"] == "wt-good"


@pytest.mark.asyncio
async def test_outer_loop_events_sse(tmp_path: Path) -> None:
    """Verify SSE stream from /outer-loop/events."""
    ol_dir = tmp_path / ".factory" / "outer_loop"
    ol_dir.mkdir(parents=True)
    events_file = ol_dir / "events.jsonl"
    ev1 = {"generation": 0, "best_score": 0.5, "mean_score": 0.3,
           "diversity": 0.02, "archive_size": 4}
    ev2 = {"generation": 1, "best_score": 0.7, "mean_score": 0.4,
           "diversity": 0.03, "archive_size": 6}
    events_file.write_text(json.dumps(ev1) + "\n" + json.dumps(ev2) + "\n")

    app = create_app(project_root=tmp_path)
    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
    collected: list[dict] = []

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        try:
            async with asyncio.timeout(5.0):
                async with c.stream("GET", "/outer-loop/events") as resp:
                    assert resp.status_code == 200
                    assert "text/event-stream" in resp.headers["content-type"]
                    async for line in resp.aiter_lines():
                        if line.startswith("data:"):
                            data_str = line[len("data:"):].strip()
                            collected.append(json.loads(data_str))
        except TimeoutError:
            pass

    assert len(collected) >= 2
    assert collected[0]["generation"] == 0
    assert collected[0]["best_score"] == 0.5
    assert collected[1]["generation"] == 1


@pytest.mark.asyncio
async def test_dashboard_html_contains_chess_elements(
    client: httpx.AsyncClient,
) -> None:
    """Verify the HTML response contains key chess UI identifiers."""
    resp = await client.get("/")
    assert resp.status_code == 200
    html = resp.text
    # Board-related elements
    assert "boards-container" in html
    assert "board-card" in html
    assert "renderBoard" in html
    # Mutation bars
    assert "mutation-bars" in html
    assert "mut-bar" in html
    # Score chart
    assert "chart-container" in html
    assert "uPlot" in html
    # SVG board rendering
    assert "parseFEN" in html
    assert "PIECE_MAP" in html
    # Eval bar
    assert "eval-bar" in html
    assert "eval_curve" in html
