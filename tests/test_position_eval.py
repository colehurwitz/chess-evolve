"""Tests for position_eval — dry-run mode and mocked LLM pipeline.

No live LLM, no live Stockfish — all external deps are mocked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import chess
import chess.engine
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
POSITIONS_FILE = REPO_ROOT / "eval" / "test_positions.json"


@pytest.fixture(autouse=True)
def _positions_env(monkeypatch):
    monkeypatch.setenv("CHESS_POSITIONS_FILE", str(POSITIONS_FILE))


# ── Fake Stockfish engine ────────────────────────────────────────


class _FakeEngine:
    """Returns pre-canned analyse() results; ignores quit()."""

    def __init__(self):
        self._i = 0

    def analyse(self, board, limit):  # noqa: ARG002
        # Return +50cp from White's perspective consistently.
        # This ensures that when verify() negates the after-move score
        # (opponent's POV), the chosen_value matches best_value → CPL ≈ 0.
        return {
            "score": chess.engine.PovScore(
                chess.engine.Cp(50), chess.WHITE,
            ),
            "pv": [next(iter(board.legal_moves))],
        }

    def quit(self):
        pass


def _patch_stockfish(monkeypatch):
    """Patch resolve_stockfish and SimpleEngine.popen_uci."""
    monkeypatch.setattr(
        "chess_evolve.tasks.resolve_stockfish", lambda: "/fake/stockfish",
    )
    monkeypatch.setattr(
        chess.engine.SimpleEngine, "popen_uci",
        lambda *a, **k: _FakeEngine(),
    )


# ── Dry-run mode tests ──────────────────────────────────────────


class TestDryRun:
    @pytest.mark.asyncio
    async def test_dry_run_returns_valid_structure(self, monkeypatch):
        """dry_run=True skips LLM, verify() reports 'no move parsed'."""
        _patch_stockfish(monkeypatch)
        from chess_evolve.position_eval import run_position_eval

        result = await run_position_eval(dry_run=True)

        assert isinstance(result, dict)
        assert "per_instance" in result
        assert "mean_cpl" in result
        assert "mean_score" in result
        assert "pass_rate" in result
        assert "passes" in result
        assert "count" in result
        assert "results_path" in result
        assert "halted" in result
        assert "halt_reason" in result
        assert result["count"] > 0
        assert result["halted"] is False
        # In dry-run, no moves are generated → all should fail to parse
        for item in result["per_instance"]:
            assert item["move"] is None

    @pytest.mark.asyncio
    async def test_dry_run_does_not_call_pipeline(self, monkeypatch):
        """Verify the LLM pipeline is never invoked in dry-run mode."""
        _patch_stockfish(monkeypatch)
        from chess_evolve import position_eval

        mock_pipeline = AsyncMock(return_value=True)
        monkeypatch.setattr(
            position_eval, "_run_pipeline_for_position", mock_pipeline,
        )

        await position_eval.run_position_eval(dry_run=True)
        mock_pipeline.assert_not_called()


# ── Mocked LLM pipeline test ────────────────────────────────────


class TestMockedPipeline:
    @pytest.mark.asyncio
    async def test_pipeline_produces_nonzero_scores(self, monkeypatch):
        """With a mocked agent that returns a legal move, scores are non-zero."""
        _patch_stockfish(monkeypatch)
        from chess_evolve import position_eval

        async def _fake_pipeline(workspace: Path) -> bool:
            """Write a known legal move to move.md."""
            board_file = workspace / ".factory" / "chess" / "board_state.md"
            content = board_file.read_text()
            # Extract FEN from the board_state.md
            for line in content.splitlines():
                if line.startswith("FEN:"):
                    fen = line.split("FEN:", 1)[1].strip()
                    break
            else:
                return False
            board = chess.Board(fen)
            if board.is_game_over():
                return True
            # Pick the first legal move
            move = next(iter(board.legal_moves)).uci()
            move_file = workspace / ".factory" / "chess" / "move.md"
            move_file.write_text(move)
            return True

        monkeypatch.setattr(
            position_eval, "_run_pipeline_for_position", _fake_pipeline,
        )

        result = await position_eval.run_position_eval(dry_run=False)

        assert result["count"] > 0
        # At least some positions should have a move parsed
        moves_found = sum(
            1 for item in result["per_instance"] if item["move"] is not None
        )
        assert moves_found > 0
        # Mean score should be > 0 (not all-zeros)
        assert result["mean_score"] > 0.0
