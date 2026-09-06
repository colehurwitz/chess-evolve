"""End-to-end tests for ChessEvolveTask with real Stockfish.

These tests play actual games against Stockfish — no mocks.
Skipped automatically if Stockfish is not installed.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

from chess_evolve.config import STOCKFISH_PATH
from chess_evolve.task import ChessEvolveTask, TaskInstance, VerifyResult

_stockfish_available = os.path.exists(STOCKFISH_PATH)
_skip_no_stockfish = pytest.mark.skipif(
    not _stockfish_available,
    reason=f"Stockfish not found at {STOCKFISH_PATH}",
)


class TestInstances:
    def test_instances_returns_elo_identifiers(self):
        task = ChessEvolveTask()
        instances = list(task.instances())
        ids = [inst.id for inst in instances]
        assert "elo_1320" in ids
        assert "elo_1420" in ids
        assert "elo_1520" in ids
        assert "elo_1620" in ids


@_skip_no_stockfish
class TestVerifyRealGame:
    @pytest.mark.timeout(300)
    def test_verify_real_game(self, tmp_path: Path):
        task = ChessEvolveTask()
        instance = TaskInstance(
            id="elo_1320",
            metadata={"opponent_elo": 1320, "games_per_eval": 1},
        )
        task.setup(instance, tmp_path)
        result = task.verify(instance, tmp_path)

        assert isinstance(result, VerifyResult)
        assert isinstance(result.score, float)
        assert math.isfinite(result.score)

        expected_keys = {
            "wins", "draws", "losses", "total_moves", "composite_score",
            "games", "avg_eval", "blunder_count", "score", "win_rate",
            "total_errors", "total_pipeline_runs",
        }
        assert expected_keys.issubset(result.details.keys())

        games = result.details["games"]
        assert isinstance(games, list)
        assert len(games) > 0

        for game in games:
            assert "result" in game
            assert "eval_curve" in game
            assert "move_list" in game
            assert game["result"] in {"win", "draw", "loss"}
            assert isinstance(game["eval_curve"], list)
            assert isinstance(game["move_list"], list)


class TestVerifyResultSchema:
    def test_rejects_unknown_fields(self):
        with pytest.raises(Exception):
            VerifyResult(passed=True, score=1.0, details={}, unknown_field="bad")

    def test_enforces_types(self):
        with pytest.raises(Exception):
            VerifyResult(passed="not_a_bool", score=1.0, details={})

    def test_valid_construction(self):
        result = VerifyResult(passed=True, score=42.5, details={"key": "value"})
        assert result.passed is True
        assert result.score == 42.5
        assert result.details == {"key": "value"}
