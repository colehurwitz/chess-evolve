"""Tests for ChessEvolveTask — verifies the Task abstraction produces
equivalent data to the existing EvalResult-based evaluation path."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from factory.task import TaskInstance, VerifyResult

from chess_evolve.config import ELO_OPTIONS, GAMES_PER_EVAL, MAX_MOVES
from chess_evolve.game import EvalResult
from chess_evolve.task import ChessEvolveTask

# ── Fixtures ────────────────────────────────────────────────────


def _make_mock_eval_result() -> EvalResult:
    """Build a realistic EvalResult with representative game data."""
    games = [
        {
            "result": "win",
            "score": 0,
            "moves": 32,
            "llm_errors": 1,
            "pipeline_runs": 16,
            "termination": "CHECKMATE",
            "move_list": ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"],
            "eval_curve": [50, 80, 120, 200, 350, 500],
            "agent_outputs": {
                "generator": ["e2e4", "g1f3", "f1b5"],
                "ranker": ["e2e4"],
                "verifier": ["e2e4"],
            },
            "illegal_moves": ["move 3: selector=z9z9 (not legal)"],
        },
        {
            "result": "draw",
            "score": 0,
            "moves": 50,
            "llm_errors": 2,
            "pipeline_runs": 25,
            "termination": "STALEMATE",
            "move_list": ["d2d4", "d7d5", "c2c4", "e7e6"],
            "eval_curve": [10, 20, -5, 0],
            "agent_outputs": {"generator": ["d2d4", "c2c4"]},
            "illegal_moves": [],
        },
        {
            "result": "loss",
            "score": 0,
            "moves": 28,
            "llm_errors": 5,
            "pipeline_runs": 14,
            "termination": "CHECKMATE",
            "move_list": ["e2e4", "e7e5", "f2f3", "d8h4"],
            "eval_curve": [50, 30, -100, -500],
            "agent_outputs": {},
            "illegal_moves": [
                "move 2: selector=a1a8 (not legal)",
                "move 4: selector=h1h8 (not legal)",
            ],
        },
    ]
    result = EvalResult(
        wins=1,
        draws=1,
        losses=1,
        total_moves=110,
        total_errors=8,
        total_pipeline_runs=55,
        games=games,
    )
    return result


# ── Test: instances() ────────────────────────────────────────────


class TestChessEvolveTaskInstances:
    def test_yields_one_instance_per_elo(self):
        task = ChessEvolveTask()
        instances = list(task.instances())
        assert len(instances) == len(ELO_OPTIONS)

    def test_instance_ids_match_elo_options(self):
        task = ChessEvolveTask()
        instances = list(task.instances())
        expected_ids = [f"elo_{elo}" for elo in ELO_OPTIONS]
        actual_ids = [inst.id for inst in instances]
        assert actual_ids == expected_ids

    def test_instance_metadata_contains_opponent_elo(self):
        task = ChessEvolveTask()
        instances = list(task.instances())
        for inst, elo in zip(instances, ELO_OPTIONS):
            assert inst.metadata["opponent_elo"] == elo

    def test_instance_metadata_contains_games_per_eval(self):
        task = ChessEvolveTask()
        instances = list(task.instances())
        for inst in instances:
            assert inst.metadata["games_per_eval"] == GAMES_PER_EVAL

    def test_instances_are_task_instance_type(self):
        task = ChessEvolveTask()
        for inst in task.instances():
            assert isinstance(inst, TaskInstance)


# ── Test: setup() ────────────────────────────────────────────────


class TestChessEvolveTaskSetup:
    @patch("chess_evolve.task.setup_workspace")
    def test_setup_calls_setup_workspace(self, mock_setup: MagicMock):
        task = ChessEvolveTask()
        workspace = Path("/tmp/test-workspace")
        instance = TaskInstance(id="elo_1320", metadata={"opponent_elo": 1320})
        task.setup(instance, workspace)
        mock_setup.assert_called_once_with(workspace)


# ── Test: prompt() ───────────────────────────────────────────────


class TestChessEvolveTaskPrompt:
    def test_prompt_is_not_the_generic_base_prompt(self):
        """The base Task.prompt() returns a generic placeholder — the chess
        task must override it with domain-specific instructions."""
        task = ChessEvolveTask()
        instance = TaskInstance(id="elo_1320", metadata={"opponent_elo": 1320})
        assert task.prompt(instance) != "Implement the feature. All tests must pass."

    def test_prompt_mentions_instance_opponent_elo(self):
        task = ChessEvolveTask()
        for elo in ELO_OPTIONS:
            instance = TaskInstance(id=f"elo_{elo}", metadata={"opponent_elo": elo})
            assert str(elo) in task.prompt(instance)

    def test_prompt_differs_per_instance(self):
        task = ChessEvolveTask()
        prompts = {task.prompt(inst) for inst in task.instances()}
        assert len(prompts) == len(ELO_OPTIONS)

    def test_prompt_mentions_games_per_eval(self):
        task = ChessEvolveTask()
        instance = TaskInstance(
            id="elo_1520", metadata={"opponent_elo": 1520, "games_per_eval": 7}
        )
        assert "7" in task.prompt(instance)

    def test_prompt_defaults_games_per_eval(self):
        task = ChessEvolveTask()
        instance = TaskInstance(
            id="elo_1320",
            metadata={"opponent_elo": 1320},  # no games_per_eval key
        )
        assert str(GAMES_PER_EVAL) in task.prompt(instance)

    def test_prompt_survives_instance_without_metadata(self):
        """Task.run() may hand over a bare instance; prompt() must not raise."""
        task = ChessEvolveTask()
        text = task.prompt(TaskInstance(id="default"))
        assert isinstance(text, str)
        assert text.strip()

    def test_prompt_describes_move_artifact_contract(self):
        task = ChessEvolveTask()
        text = task.prompt(TaskInstance(id="elo_1320", metadata={"opponent_elo": 1320}))
        assert ".factory/chess/board_state.md" in text
        assert ".factory/chess/move.md" in text
        assert "UCI" in text

    def test_prompt_names_the_mutable_surfaces(self):
        task = ChessEvolveTask()
        text = task.prompt(TaskInstance(id="elo_1320", metadata={"opponent_elo": 1320}))
        assert "src/chess_evolve/prompts.py" in text
        assert "src/chess_evolve/pipeline.py" in text

    def test_prompt_explains_scoring(self):
        task = ChessEvolveTask()
        text = task.prompt(TaskInstance(id="elo_1320", metadata={"opponent_elo": 1320}))
        assert "composite_score" in text
        assert str(MAX_MOVES) in text

    def test_prompt_has_no_unresolved_placeholders(self):
        task = ChessEvolveTask()
        for inst in task.instances():
            text = task.prompt(inst)
            assert "{" not in text and "}" not in text


# ── Test: verify() ───────────────────────────────────────────────


class TestChessEvolveTaskVerify:
    @patch("chess_evolve.task.build_pipeline")
    @patch("chess_evolve.task.evaluate_pipeline", new_callable=AsyncMock)
    def test_verify_returns_verify_result(
        self, mock_eval: AsyncMock, mock_build: MagicMock
    ):
        mock_result = _make_mock_eval_result()
        mock_eval.return_value = mock_result
        mock_build.return_value = MagicMock()

        task = ChessEvolveTask()
        instance = TaskInstance(
            id="elo_1320",
            metadata={"opponent_elo": 1320, "games_per_eval": 3},
        )
        result = task.verify(instance, Path("/tmp/test-workspace"))

        assert isinstance(result, VerifyResult)

    @patch("chess_evolve.task.build_pipeline")
    @patch("chess_evolve.task.evaluate_pipeline", new_callable=AsyncMock)
    def test_verify_passed_with_wins(
        self, mock_eval: AsyncMock, mock_build: MagicMock
    ):
        mock_result = _make_mock_eval_result()
        mock_eval.return_value = mock_result
        mock_build.return_value = MagicMock()

        task = ChessEvolveTask()
        instance = TaskInstance(
            id="elo_1320",
            metadata={"opponent_elo": 1320, "games_per_eval": 3},
        )
        result = task.verify(instance, Path("/tmp/test-workspace"))

        # wins=1, draws=1 -> (1 + 1) > 0 -> True
        assert result.passed is True

    @patch("chess_evolve.task.build_pipeline")
    @patch("chess_evolve.task.evaluate_pipeline", new_callable=AsyncMock)
    def test_verify_passed_false_all_losses(
        self, mock_eval: AsyncMock, mock_build: MagicMock
    ):
        mock_result = EvalResult(
            wins=0, draws=0, losses=3, total_moves=30,
            games=[{"eval_curve": [-100, -200, -300]}],
        )
        mock_eval.return_value = mock_result
        mock_build.return_value = MagicMock()

        task = ChessEvolveTask()
        instance = TaskInstance(
            id="elo_1520",
            metadata={"opponent_elo": 1520, "games_per_eval": 3},
        )
        result = task.verify(instance, Path("/tmp/test-workspace"))

        assert result.passed is False

    @patch("chess_evolve.task.build_pipeline")
    @patch("chess_evolve.task.evaluate_pipeline", new_callable=AsyncMock)
    def test_verify_score_equals_composite_score(
        self, mock_eval: AsyncMock, mock_build: MagicMock
    ):
        mock_result = _make_mock_eval_result()
        mock_eval.return_value = mock_result
        mock_build.return_value = MagicMock()

        task = ChessEvolveTask()
        instance = TaskInstance(
            id="elo_1320",
            metadata={"opponent_elo": 1320, "games_per_eval": 3},
        )
        result = task.verify(instance, Path("/tmp/test-workspace"))

        assert result.score == mock_result.composite_score

    @patch("chess_evolve.task.build_pipeline")
    @patch("chess_evolve.task.evaluate_pipeline", new_callable=AsyncMock)
    def test_verify_details_mirrors_eval_result(
        self, mock_eval: AsyncMock, mock_build: MagicMock
    ):
        """Core equivalence test: VerifyResult.details must contain ALL
        EvalResult data with EXACT same keys and values."""
        mock_result = _make_mock_eval_result()
        mock_eval.return_value = mock_result
        mock_build.return_value = MagicMock()

        task = ChessEvolveTask()
        instance = TaskInstance(
            id="elo_1320",
            metadata={"opponent_elo": 1320, "games_per_eval": 3},
        )
        result = task.verify(instance, Path("/tmp/test-workspace"))
        details = result.details

        # Scalar fields
        assert details["wins"] == mock_result.wins
        assert details["draws"] == mock_result.draws
        assert details["losses"] == mock_result.losses
        assert details["total_moves"] == mock_result.total_moves
        assert details["total_errors"] == mock_result.total_errors
        assert details["total_pipeline_runs"] == mock_result.total_pipeline_runs

        # Computed properties
        assert details["score"] == mock_result.score
        assert details["avg_eval"] == mock_result.avg_eval
        assert details["blunder_count"] == mock_result.blunder_count
        assert details["composite_score"] == mock_result.composite_score
        assert details["win_rate"] == mock_result.win_rate

        # Games list — same objects, same keys per game
        assert details["games"] is mock_result.games
        assert len(details["games"]) == 3
        for game in details["games"]:
            assert "result" in game
            assert "score" in game
            assert "moves" in game
            assert "llm_errors" in game
            assert "pipeline_runs" in game
            assert "termination" in game
            assert "move_list" in game
            assert "eval_curve" in game
            assert "agent_outputs" in game
            assert "illegal_moves" in game

    @patch("chess_evolve.task.build_pipeline")
    @patch("chess_evolve.task.evaluate_pipeline", new_callable=AsyncMock)
    def test_verify_uses_correct_opponent_elo(
        self, mock_eval: AsyncMock, mock_build: MagicMock
    ):
        mock_eval.return_value = _make_mock_eval_result()
        mock_pipeline = MagicMock()
        mock_build.return_value = mock_pipeline

        task = ChessEvolveTask()
        instance = TaskInstance(
            id="elo_1620",
            metadata={"opponent_elo": 1620, "games_per_eval": 2},
        )
        task.verify(instance, Path("/tmp/test-workspace"))

        # build_pipeline should receive a config with the right ELO
        call_args = mock_build.call_args
        cfg = call_args[0][0]
        assert cfg.opponent_elo == 1620

        # evaluate_pipeline should receive the correct n_games
        eval_call = mock_eval.call_args
        assert eval_call[1]["n_games"] == 2

    @patch("chess_evolve.task.build_pipeline")
    @patch("chess_evolve.task.evaluate_pipeline", new_callable=AsyncMock)
    def test_verify_defaults_games_per_eval(
        self, mock_eval: AsyncMock, mock_build: MagicMock
    ):
        mock_eval.return_value = _make_mock_eval_result()
        mock_build.return_value = MagicMock()

        task = ChessEvolveTask()
        instance = TaskInstance(
            id="elo_1320",
            metadata={"opponent_elo": 1320},  # no games_per_eval key
        )
        task.verify(instance, Path("/tmp/test-workspace"))

        eval_call = mock_eval.call_args
        assert eval_call[1]["n_games"] == GAMES_PER_EVAL


# ── Test: task name ──────────────────────────────────────────────


class TestChessEvolveTaskName:
    def test_task_name(self):
        task = ChessEvolveTask()
        assert task.name == "chess-evolve"
