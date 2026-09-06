"""End-to-end test for ChessEvolveTask.verify() against a real Stockfish engine.

Skipped automatically when Stockfish is not installed.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from factory.task import TaskInstance, VerifyResult

from chess_evolve.config import STOCKFISH_PATH
from chess_evolve.task import ChessEvolveTask

_stockfish_available = (
    shutil.which("stockfish") is not None or Path(STOCKFISH_PATH).exists()
)

EXPECTED_DETAIL_KEYS = {
    "wins",
    "draws",
    "losses",
    "total_moves",
    "total_errors",
    "total_pipeline_runs",
    "games",
    "score",
    "avg_eval",
    "blunder_count",
    "composite_score",
    "win_rate",
}


@pytest.mark.skipif(not _stockfish_available, reason="Stockfish not installed")
def test_verify_elo_1320_returns_complete_result(tmp_path: Path) -> None:
    instance = TaskInstance(
        id="elo_1320",
        metadata={"opponent_elo": 1320, "games_per_eval": 1},
    )
    task = ChessEvolveTask()
    result = task.verify(instance, tmp_path)

    assert isinstance(result, VerifyResult)
    assert isinstance(result.passed, bool)
    assert isinstance(result.score, float)

    details = result.details
    assert set(details.keys()) == EXPECTED_DETAIL_KEYS

    assert isinstance(details["wins"], int) and details["wins"] >= 0
    assert isinstance(details["draws"], int) and details["draws"] >= 0
    assert isinstance(details["losses"], int) and details["losses"] >= 0
    assert details["wins"] + details["draws"] + details["losses"] == 1

    assert isinstance(details["total_moves"], int) and details["total_moves"] > 0
    assert isinstance(details["total_errors"], int) and details["total_errors"] >= 0
    assert isinstance(details["total_pipeline_runs"], int) and details["total_pipeline_runs"] >= 0

    assert isinstance(details["games"], list) and len(details["games"]) > 0

    assert isinstance(details["score"], (int, float))
    assert isinstance(details["avg_eval"], (int, float))
    assert isinstance(details["blunder_count"], int) and details["blunder_count"] >= 0
    assert isinstance(details["composite_score"], float)
    assert isinstance(details["win_rate"], str)
