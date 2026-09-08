"""End-to-end tests for ChessEvolveTask.verify() against a real Stockfish game.

tests/test_task.py mocks ``evaluate_pipeline`` and asserts the VerifyResult
mirrors a synthetic EvalResult. These tests drive the real stack instead:

    ChessEvolveTask.verify()
      -> build_pipeline()          (real Package / Workflow)
      -> evaluate_pipeline()       (real)
      -> play_game()               (real)
      -> Stockfish                 (two real UCI processes: opponent + evaluator)

Only the LLM boundary (``chess_evolve.engine._api_call``) is stubbed, so the
WorkflowExecutor, the legality gate subprocess, move extraction, the eval
curve and the result bookkeeping are all exercised for real. The stub reads
the "Legal moves:" line out of the prompt the pipeline writes, which makes the
game deterministic and keeps the suite fast and free.

``test_verify_live_llm`` additionally uses the real LLM. It is opt-in via
CHESS_E2E_LIVE=1 because it costs money and takes minutes.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

import pytest
from factory.task import VerifyResult

from chess_evolve import broadcast, engine, game
from chess_evolve.config import ELO_OPTIONS
from chess_evolve.task import ChessEvolveTask

# The 12 fields ChessEvolveTask.verify() promises in VerifyResult.details.
EXPECTED_DETAIL_FIELDS = (
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
)

# Every key play_game() puts on a per-game dict.
EXPECTED_GAME_FIELDS = (
    "result",
    "score",
    "moves",
    "llm_errors",
    "pipeline_runs",
    "termination",
    "move_list",
    "eval_curve",
    "agent_outputs",
    "illegal_moves",
)

E2E_ELO = 1320
E2E_INSTANCE_ID = f"elo_{E2E_ELO}"

# A full 60-ply game is slower than CI's --timeout=30 allows. 16 plies is
# enough to exercise every code path an end-to-end run touches.
E2E_MAX_MOVES = 16

LEGAL_MOVES_RE = re.compile(r"^Legal moves:\s*(.+)$", re.MULTILINE)


def _stockfish_path() -> str | None:
    """Return a usable Stockfish binary, or None if the host has none."""
    if Path(game.STOCKFISH_PATH).exists():
        return game.STOCKFISH_PATH
    return shutil.which("stockfish")


requires_stockfish = pytest.mark.skipif(
    _stockfish_path() is None,
    reason="Stockfish binary not installed",
)


async def _stub_api_call(
    system_prompt: str, user_msg: str, max_tokens: int = 200,
) -> str:
    """Stand in for the LLM: answer with the first legal move in the prompt.

    ``_sdk_invoke_agent`` appends board_state.md to the node's task, so the
    prompt always carries the "Legal moves: e2e4, d2d4, ..." line that
    ``engine.write_board_state`` / ``_board_user_msg`` produce. Replying with
    a real UCI move keeps the legality gate on its PROCEED path, which is the
    branch a live run normally takes.
    """
    match = LEGAL_MOVES_RE.search(user_msg)
    if not match:
        return ""
    moves = [m.strip() for m in match.group(1).split(",") if m.strip()]
    return moves[0] if moves else ""


@pytest.fixture(scope="module")
def e2e_result(tmp_path_factory: pytest.TempPathFactory) -> VerifyResult:
    """Run ChessEvolveTask.verify() once for real, and share the result.

    Module-scoped: one Stockfish game covers every assertion below, and
    playing it per-test would multiply the suite's runtime for no extra
    coverage.
    """
    stockfish = _stockfish_path()
    if stockfish is None:  # pragma: no cover - guarded by requires_stockfish
        pytest.skip("Stockfish binary not installed")

    workspace = tmp_path_factory.mktemp("chess-e2e")

    with pytest.MonkeyPatch.context() as mp:
        # Keep the game off the real /tmp/chess-factory workspace.
        mp.setattr(game, "WORKSPACE", workspace / "games")
        mp.setattr(broadcast, "LIVE_DIR", workspace / "live")
        mp.setattr(game, "STOCKFISH_PATH", stockfish)
        mp.setattr(game, "MAX_MOVES", E2E_MAX_MOVES)
        # No LLM calls: reply locally with a legal move.
        mp.setattr(engine, "_api_call", _stub_api_call)
        # The legality gate shells out to `python3 -c "import chess; ..."`.
        # Put this interpreter first so the gate finds python-chess.
        mp.setenv("PATH", f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}")

        task = ChessEvolveTask()
        instance = next(i for i in task.instances() if i.id == E2E_INSTANCE_ID)
        return task.verify(instance, workspace)


# ── Test: the 12-field details contract ──────────────────────────


@requires_stockfish
class TestVerifyDetailsContract:
    def test_returns_verify_result(self, e2e_result: VerifyResult):
        assert isinstance(e2e_result, VerifyResult)

    def test_details_contains_all_twelve_fields(self, e2e_result: VerifyResult):
        assert len(EXPECTED_DETAIL_FIELDS) == 12
        missing = set(EXPECTED_DETAIL_FIELDS) - set(e2e_result.details)
        assert not missing, f"VerifyResult.details is missing {sorted(missing)}"

    def test_details_has_no_unexpected_fields(self, e2e_result: VerifyResult):
        extra = set(e2e_result.details) - set(EXPECTED_DETAIL_FIELDS)
        assert not extra, f"VerifyResult.details grew {sorted(extra)}"

    def test_detail_field_types(self, e2e_result: VerifyResult):
        details = e2e_result.details
        assert isinstance(details["wins"], int)
        assert isinstance(details["draws"], int)
        assert isinstance(details["losses"], int)
        assert isinstance(details["total_moves"], int)
        assert isinstance(details["total_errors"], int)
        assert isinstance(details["total_pipeline_runs"], int)
        assert isinstance(details["games"], list)
        assert isinstance(details["score"], float)
        assert isinstance(details["avg_eval"], float)
        assert isinstance(details["blunder_count"], int)
        assert isinstance(details["composite_score"], float)
        assert isinstance(details["win_rate"], str)


# ── Test: the game actually happened ─────────────────────────────


@requires_stockfish
class TestVerifyPlayedARealGame:
    def test_one_game_per_games_per_eval(self, e2e_result: VerifyResult):
        details = e2e_result.details
        assert len(details["games"]) == 1
        assert details["wins"] + details["draws"] + details["losses"] == 1

    def test_every_game_carries_the_full_record(self, e2e_result: VerifyResult):
        for game_record in e2e_result.details["games"]:
            missing = set(EXPECTED_GAME_FIELDS) - set(game_record)
            assert not missing, f"game record is missing {sorted(missing)}"

    def test_moves_were_played(self, e2e_result: VerifyResult):
        details = e2e_result.details
        assert details["total_moves"] > 0
        assert details["total_moves"] <= E2E_MAX_MOVES
        for game_record in details["games"]:
            assert game_record["move_list"], "no moves recorded"
            assert len(game_record["move_list"]) == game_record["moves"]

    def test_move_list_is_uci(self, e2e_result: VerifyResult):
        uci = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")
        for game_record in e2e_result.details["games"]:
            for move in game_record["move_list"]:
                assert uci.match(move), f"{move!r} is not a UCI move"

    def test_stockfish_produced_an_eval_curve(self, e2e_result: VerifyResult):
        for game_record in e2e_result.details["games"]:
            curve = game_record["eval_curve"]
            assert curve, "Stockfish returned no evaluations"
            assert len(curve) == len(game_record["move_list"])
            assert all(isinstance(cp, int) for cp in curve)

    def test_pipeline_ran_for_every_llm_move(self, e2e_result: VerifyResult):
        for game_record in e2e_result.details["games"]:
            assert game_record["pipeline_runs"] > 0
            # One pipeline run per LLM move; the LLM has white and so moves on
            # every other ply, taking the extra ply on odd-length games.
            expected = (game_record["moves"] + 1) // 2
            assert game_record["pipeline_runs"] == expected

    def test_result_is_a_known_outcome(self, e2e_result: VerifyResult):
        for game_record in e2e_result.details["games"]:
            assert game_record["result"] in {"win", "draw", "loss"}
        assert e2e_result.details["win_rate"].startswith("+")


# ── Test: derived scores agree with the raw data ─────────────────


@requires_stockfish
class TestVerifyScoring:
    def test_score_is_the_composite_score(self, e2e_result: VerifyResult):
        assert e2e_result.score == e2e_result.details["composite_score"]

    def test_passed_tracks_wins_and_draws(self, e2e_result: VerifyResult):
        details = e2e_result.details
        expected = (details["wins"] + details["draws"]) > 0
        assert e2e_result.passed is expected

    def test_avg_eval_matches_the_curve(self, e2e_result: VerifyResult):
        curve: list[int] = []
        for game_record in e2e_result.details["games"]:
            curve.extend(game_record["eval_curve"])
        assert e2e_result.details["avg_eval"] == pytest.approx(
            sum(curve) / len(curve)
        )

    def test_blunder_count_matches_the_curve(self, e2e_result: VerifyResult):
        expected = 0
        for game_record in e2e_result.details["games"]:
            curve = game_record["eval_curve"]
            expected += sum(
                1 for i in range(1, len(curve)) if curve[i] - curve[i - 1] < -200
            )
        assert e2e_result.details["blunder_count"] == expected

    def test_win_rate_matches_the_counts(self, e2e_result: VerifyResult):
        details = e2e_result.details
        assert details["win_rate"] == (
            f"+{details['wins']}={details['draws']}-{details['losses']}"
        )


# ── Test: the instance under test ────────────────────────────────


class TestE2EInstance:
    def test_elo_1320_is_a_real_instance(self):
        task = ChessEvolveTask()
        instance = next(
            (i for i in task.instances() if i.id == E2E_INSTANCE_ID), None
        )
        assert instance is not None
        assert instance.metadata["opponent_elo"] == E2E_ELO
        assert E2E_ELO in ELO_OPTIONS


# ── Test: live LLM (opt-in) ──────────────────────────────────────


@requires_stockfish
@pytest.mark.skipif(
    os.environ.get("CHESS_E2E_LIVE", "").lower() not in ("1", "true", "yes"),
    reason="live LLM run: set CHESS_E2E_LIVE=1 to enable",
)
def test_verify_live_llm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The same contract, with no stub at all — real LLM, real Stockfish."""
    monkeypatch.setattr(game, "WORKSPACE", tmp_path / "games")
    monkeypatch.setattr(broadcast, "LIVE_DIR", tmp_path / "live")
    monkeypatch.setattr(game, "STOCKFISH_PATH", _stockfish_path())
    monkeypatch.setenv(
        "PATH", f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"
    )

    task = ChessEvolveTask()
    instance = next(i for i in task.instances() if i.id == E2E_INSTANCE_ID)
    result = task.verify(instance, tmp_path)

    assert isinstance(result, VerifyResult)
    missing = set(EXPECTED_DETAIL_FIELDS) - set(result.details)
    assert not missing, f"VerifyResult.details is missing {sorted(missing)}"
    assert result.details["total_moves"] > 0
    assert result.score == result.details["composite_score"]
