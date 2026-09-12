"""Hermetic tests for GameTask, advance_game_state, and game-eval workflow.

No live LLM, no live Stockfish — all external dependencies are mocked via
``unittest.mock.patch`` (no monkey-patching).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import chess
import chess.engine
import pytest

from chess_evolve.config import ELO_OPTIONS
from chess_evolve.tasks import GameTask, _extract_blunders

# ── Helpers ──────────────────────────────────────────────────────


def _write_game_state(workspace: Path, state: dict) -> Path:
    """Write a canned game_state.json to the workspace."""
    chess_dir = workspace / ".factory" / "chess"
    chess_dir.mkdir(parents=True, exist_ok=True)
    path = chess_dir / "game_state.json"
    path.write_text(json.dumps(state))
    return path


def _cp(value: int, turn: bool) -> chess.engine.PovScore:
    return chess.engine.PovScore(chess.engine.Cp(value), turn)


class _FakePlayResult:
    """Mimics ``chess.engine.PlayResult`` returned by ``engine.play()``."""

    def __init__(self, move: chess.Move):
        self.move = move


class _FakeEngine:
    """Fake Stockfish engine for advance_game_state tests."""

    def __init__(self) -> None:
        self._configured = False

    def configure(self, options: dict) -> None:
        self._configured = True

    def play(self, board: chess.Board, limit: object) -> _FakePlayResult:
        # Just play the first legal move
        return _FakePlayResult(next(iter(board.legal_moves)))

    def analyse(
        self, board: chess.Board, limit: object,
    ) -> dict:
        return {"score": _cp(0, chess.WHITE)}

    def quit(self) -> None:
        pass


# ── TestGameTaskInstances ────────────────────────────────────────


class TestGameTaskInstances:
    def test_instance_count(self) -> None:
        instances = list(GameTask().instances())
        expected = len(ELO_OPTIONS) * 2
        assert len(instances) == expected

    def test_instance_ids(self) -> None:
        instances = list(GameTask().instances())
        for inst in instances:
            assert inst.id.startswith("elo")
            assert "_white" in inst.id or "_black" in inst.id

    def test_instance_metadata(self) -> None:
        instances = list(GameTask().instances())
        for inst in instances:
            assert isinstance(inst.metadata["opponent_elo"], int)
            assert inst.metadata["color"] in ("white", "black")


# ── TestGameTaskSetup ────────────────────────────────────────────


class TestGameTaskSetup:
    def test_creates_board_state(self, tmp_path: Path) -> None:
        task = GameTask()
        inst = next(iter(task.instances()))
        task.setup(inst, tmp_path)
        board_file = tmp_path / ".factory" / "chess" / "board_state.md"
        assert board_file.exists()
        assert chess.Board().fen() in board_file.read_text()

    def test_creates_game_state_json(self, tmp_path: Path) -> None:
        task = GameTask()
        inst = next(iter(task.instances()))
        task.setup(inst, tmp_path)
        state_path = tmp_path / ".factory" / "chess" / "game_state.json"
        assert state_path.exists()
        state = json.loads(state_path.read_text())
        assert state["opponent_elo"] == inst.metadata["opponent_elo"]
        assert state["color"] == inst.metadata["color"]
        assert state["fen"] == chess.Board().fen()
        assert state["move_list"] == []
        assert state["eval_curve"] == []
        assert state["illegal_attempts"] == 0
        assert state["game_over"] is False
        assert state["result"] is None


# ── TestGameTaskPrompt ───────────────────────────────────────────


class TestGameTaskPrompt:
    def test_prompt_contains_fen_and_uci(self) -> None:
        task = GameTask()
        inst = next(iter(task.instances()))
        text = task.prompt(inst)
        assert text  # non-empty
        assert chess.Board().fen() in text
        assert "UCI" in text


# ── TestGameTaskVerify ───────────────────────────────────────────


class TestGameTaskVerify:
    def _make_instance(
        self, elo: int = 1500, color: str = "white",
    ):
        from factory.task import TaskInstance  # noqa: PLC0415

        return TaskInstance(
            id=f"elo{elo}_{color}",
            metadata={"opponent_elo": elo, "color": color},
        )

    def test_win_passes(self, tmp_path: Path) -> None:
        state = {
            "opponent_elo": 1500,
            "color": "white",
            "fen": chess.Board().fen(),
            "move_list": ["e2e4", "e7e5", "d2d4"],
            "eval_curve": [50, 80, 120],
            "illegal_attempts": 0,
            "move_count": 3,
            "result": "win",
            "game_over": True,
        }
        _write_game_state(tmp_path, state)
        result = GameTask().verify(self._make_instance(), tmp_path)
        assert result.passed is True
        assert result.details["result"] == "win"

    def test_draw_passes(self, tmp_path: Path) -> None:
        state = {
            "opponent_elo": 1500, "color": "white",
            "fen": chess.Board().fen(),
            "move_list": ["e2e4"], "eval_curve": [10],
            "illegal_attempts": 0, "move_count": 1,
            "result": "draw", "game_over": True,
        }
        _write_game_state(tmp_path, state)
        result = GameTask().verify(self._make_instance(), tmp_path)
        assert result.passed is True

    def test_loss_fails(self, tmp_path: Path) -> None:
        state = {
            "opponent_elo": 1500, "color": "white",
            "fen": chess.Board().fen(),
            "move_list": ["e2e4"], "eval_curve": [-200],
            "illegal_attempts": 0, "move_count": 1,
            "result": "loss", "game_over": True,
        }
        _write_game_state(tmp_path, state)
        result = GameTask().verify(self._make_instance(), tmp_path)
        assert result.passed is False

    def test_composite_score(self, tmp_path: Path) -> None:
        state = {
            "opponent_elo": 1500, "color": "white",
            "fen": chess.Board().fen(),
            "move_list": ["e2e4", "e7e5", "d2d4"],
            "eval_curve": [50, 80, 120],
            "illegal_attempts": 0, "move_count": 3,
            "result": "win", "game_over": True,
        }
        _write_game_state(tmp_path, state)
        result = GameTask().verify(self._make_instance(), tmp_path)
        # position_score = (550+580+620)/1000 = 1.75
        # outcome_bonus = 200
        expected = (50 + 500) / 1000.0 + (80 + 500) / 1000.0 + (120 + 500) / 1000.0 + 200
        assert result.score == pytest.approx(expected)
        assert result.details["composite_score"] == pytest.approx(expected)

    def test_details_keys(self, tmp_path: Path) -> None:
        state = {
            "opponent_elo": 1500, "color": "black",
            "fen": chess.Board().fen(),
            "move_list": ["e2e4", "e7e5"],
            "eval_curve": [50, 30],
            "illegal_attempts": 2,
            "move_count": 2,
            "result": "draw", "game_over": True,
        }
        _write_game_state(tmp_path, state)
        result = GameTask().verify(self._make_instance(color="black"), tmp_path)
        details = result.details
        assert details["result"] == "draw"
        assert details["move_list"] == ["e2e4", "e7e5"]
        assert details["eval_curve"] == [50, 30]
        assert details["illegal_attempts"] == 2
        assert details["opponent_elo"] == 1500
        assert details["color"] == "black"
        assert details["move_count"] == 2
        assert "avg_centipawn" in details
        assert "blunders" in details
        assert "composite_score" in details

    def test_avg_centipawn(self, tmp_path: Path) -> None:
        state = {
            "opponent_elo": 1500, "color": "white",
            "fen": chess.Board().fen(),
            "move_list": ["e2e4", "e7e5"],
            "eval_curve": [100, 200],
            "illegal_attempts": 0, "move_count": 2,
            "result": "win", "game_over": True,
        }
        _write_game_state(tmp_path, state)
        result = GameTask().verify(self._make_instance(), tmp_path)
        assert result.details["avg_centipawn"] == pytest.approx(150.0)

    def test_missing_game_state(self, tmp_path: Path) -> None:
        result = GameTask().verify(self._make_instance(), tmp_path)
        assert result.passed is False
        assert result.score == 0.0
        assert result.details["error"] == "game_state_not_found"

    def test_blunders_in_details(self, tmp_path: Path) -> None:
        state = {
            "opponent_elo": 1500, "color": "white",
            "fen": chess.Board().fen(),
            "move_list": ["e2e4", "e7e5", "d2d4"],
            "eval_curve": [200, -50, 100],
            "illegal_attempts": 0, "move_count": 3,
            "result": "draw", "game_over": True,
        }
        _write_game_state(tmp_path, state)
        result = GameTask().verify(self._make_instance(), tmp_path)
        blunders = result.details["blunders"]
        assert len(blunders) == 1
        assert blunders[0]["move_index"] == 1
        assert blunders[0]["cp_drop"] == -250


# ── TestExtractBlunders ──────────────────────────────────────────


class TestExtractBlunders:
    def test_single_blunder(self) -> None:
        curve = [200, -100]
        moves = ["e2e4", "e7e5"]
        blunders = _extract_blunders(curve, moves)
        assert len(blunders) == 1
        assert blunders[0]["move_index"] == 1
        assert blunders[0]["move_num"] == 1
        assert blunders[0]["move_uci"] == "e7e5"
        assert blunders[0]["cp_before"] == 200
        assert blunders[0]["cp_after"] == -100
        assert blunders[0]["cp_drop"] == -300

    def test_smooth_curve(self) -> None:
        curve = [50, 60, 55, 70]
        moves = ["a2a3", "a7a6", "b2b3", "b7b6"]
        assert _extract_blunders(curve, moves) == []

    def test_multiple_blunders(self) -> None:
        curve = [200, -100, 300, 50]
        moves = ["a", "b", "c", "d"]
        blunders = _extract_blunders(curve, moves)
        assert len(blunders) == 2
        assert blunders[0]["move_index"] == 1
        assert blunders[1]["move_index"] == 3

    def test_empty_curve(self) -> None:
        assert _extract_blunders([], []) == []

    def test_move_list_shorter_than_curve(self) -> None:
        curve = [200, -100]
        moves = ["e2e4"]  # only 1 move, but curve has 2 entries
        blunders = _extract_blunders(curve, moves)
        assert len(blunders) == 1
        assert blunders[0]["move_uci"] == "?"


# ── TestAdvanceGameState ─────────────────────────────────────────


class TestAdvanceGameState:
    @patch("chess_evolve.engine.resolve_stockfish", return_value="/fake/sf")
    @patch.object(chess.engine.SimpleEngine, "popen_uci")
    def test_llm_turn_reloop(
        self, mock_popen: MagicMock, mock_sf: MagicMock, tmp_path: Path,
    ) -> None:
        """LLM move + Stockfish response, mid-game → prints RELOOP."""
        mock_popen.return_value = _FakeEngine()

        # Prepare workspace: LLM plays white, starting position, LLM already
        # wrote move.md with e2e4
        board = chess.Board()
        state = {
            "opponent_elo": 1500, "color": "white",
            "fen": board.fen(),
            "move_list": [], "eval_curve": [],
            "illegal_attempts": 0, "move_count": 0,
            "result": None, "game_over": False,
        }
        chess_dir = tmp_path / ".factory" / "chess"
        chess_dir.mkdir(parents=True, exist_ok=True)
        (chess_dir / "game_state.json").write_text(json.dumps(state))
        (chess_dir / "board_state.md").write_text(f"FEN: {board.fen()}\n")
        (chess_dir / "move.md").write_text("e2e4")

        from chess_evolve.engine import advance_game_state

        printed: list[str] = []

        with patch("builtins.print", side_effect=lambda *a, **kw: printed.append(
            " ".join(str(x) for x in a),
        )):
            advance_game_state(str(tmp_path))

        # Should print RELOOP (game not over)
        assert any("RELOOP" in p for p in printed)

        # game_state.json should be updated
        updated = json.loads((chess_dir / "game_state.json").read_text())
        assert len(updated["move_list"]) >= 2  # LLM + Stockfish
        assert updated["move_count"] >= 2
        assert len(updated["eval_curve"]) >= 2
        assert updated["fen"] != board.fen()  # position advanced

    @patch("chess_evolve.engine.resolve_stockfish", return_value=None)
    def test_no_stockfish_proceeds_loss(
        self, mock_sf: MagicMock, tmp_path: Path,
    ) -> None:
        """No Stockfish available → marks loss and prints PROCEED."""
        state = {
            "opponent_elo": 1500, "color": "white",
            "fen": chess.Board().fen(),
            "move_list": [], "eval_curve": [],
            "illegal_attempts": 0, "move_count": 0,
            "result": None, "game_over": False,
        }
        chess_dir = tmp_path / ".factory" / "chess"
        chess_dir.mkdir(parents=True, exist_ok=True)
        (chess_dir / "game_state.json").write_text(json.dumps(state))

        from chess_evolve.engine import advance_game_state

        printed: list[str] = []

        with patch("builtins.print", side_effect=lambda *a, **kw: printed.append(
            " ".join(str(x) for x in a),
        )):
            advance_game_state(str(tmp_path))

        assert any("PROCEED" in p for p in printed)
        updated = json.loads((chess_dir / "game_state.json").read_text())
        assert updated["result"] == "loss"
        assert updated["game_over"] is True

    @patch("chess_evolve.engine.resolve_stockfish", return_value="/fake/sf")
    @patch.object(chess.engine.SimpleEngine, "popen_uci")
    def test_black_first_move_stockfish_plays(
        self, mock_popen: MagicMock, mock_sf: MagicMock, tmp_path: Path,
    ) -> None:
        """When LLM plays black on move 1, Stockfish moves first → RELOOP."""
        mock_popen.return_value = _FakeEngine()

        board = chess.Board()
        state = {
            "opponent_elo": 1500, "color": "black",
            "fen": board.fen(),
            "move_list": [], "eval_curve": [],
            "illegal_attempts": 0, "move_count": 0,
            "result": None, "game_over": False,
        }
        chess_dir = tmp_path / ".factory" / "chess"
        chess_dir.mkdir(parents=True, exist_ok=True)
        (chess_dir / "game_state.json").write_text(json.dumps(state))
        (chess_dir / "board_state.md").write_text(f"FEN: {board.fen()}\n")

        from chess_evolve.engine import advance_game_state

        printed: list[str] = []

        with patch("builtins.print", side_effect=lambda *a, **kw: printed.append(
            " ".join(str(x) for x in a),
        )):
            advance_game_state(str(tmp_path))

        assert any("RELOOP" in p for p in printed)
        updated = json.loads((chess_dir / "game_state.json").read_text())
        # Stockfish played one move
        assert len(updated["move_list"]) == 1
        assert updated["move_count"] == 1

    @patch("chess_evolve.engine.resolve_stockfish", return_value="/fake/sf")
    @patch.object(chess.engine.SimpleEngine, "popen_uci")
    def test_illegal_move_increments_counter(
        self, mock_popen: MagicMock, mock_sf: MagicMock, tmp_path: Path,
    ) -> None:
        """When move.md contains an illegal move, illegal_attempts increments."""
        mock_popen.return_value = _FakeEngine()

        board = chess.Board()
        state = {
            "opponent_elo": 1500, "color": "white",
            "fen": board.fen(),
            "move_list": [], "eval_curve": [],
            "illegal_attempts": 0, "move_count": 0,
            "result": None, "game_over": False,
        }
        chess_dir = tmp_path / ".factory" / "chess"
        chess_dir.mkdir(parents=True, exist_ok=True)
        (chess_dir / "game_state.json").write_text(json.dumps(state))
        (chess_dir / "board_state.md").write_text(f"FEN: {board.fen()}\n")
        # Write an illegal/nonsensical move
        (chess_dir / "move.md").write_text("z9z9")

        from chess_evolve.engine import advance_game_state

        with patch("builtins.print"):
            advance_game_state(str(tmp_path))

        updated = json.loads((chess_dir / "game_state.json").read_text())
        assert updated["illegal_attempts"] == 1
        # A random legal move was played instead
        assert len(updated["move_list"]) >= 1


# ── TestBuildGameEvalWorkflow ────────────────────────────────────


class TestBuildGameEvalWorkflow:
    def test_workflow_name(self) -> None:
        from chess_evolve.pipeline import build_game_eval_workflow

        wf = build_game_eval_workflow()
        assert wf.name == "game-eval"

    def test_has_data_node(self) -> None:
        from factory.workflow.primitives import DataNode

        from chess_evolve.pipeline import GAME_TASK_REF, build_game_eval_workflow

        wf = build_game_eval_workflow()
        data_node = wf.nodes["games"]
        assert isinstance(data_node, DataNode)
        assert data_node.task_ref == GAME_TASK_REF

    def test_has_generator(self) -> None:
        from factory.workflow.primitives import AgentNode

        from chess_evolve.pipeline import build_game_eval_workflow

        wf = build_game_eval_workflow()
        # The generator should be somewhere in the compiled nodes
        gen_nodes = [
            nid for nid, n in wf.nodes.items()
            if isinstance(n, AgentNode) and "generator" in nid
        ]
        assert len(gen_nodes) >= 1

    def test_has_game_gate(self) -> None:
        from factory.workflow.primitives import GateNode

        from chess_evolve.pipeline import build_game_eval_workflow

        wf = build_game_eval_workflow()
        gate_nodes = [
            nid for nid, n in wf.nodes.items()
            if isinstance(n, GateNode) and "game_gate" in nid
        ]
        assert len(gate_nodes) >= 1

    def test_subgraph_entry_valid(self) -> None:
        from chess_evolve.pipeline import build_game_eval_workflow

        wf = build_game_eval_workflow()
        data_node = wf.nodes["games"]
        # The subgraph_entry should reference a node in the workflow
        # (may be prefixed by Loop compilation)
        found = any(
            data_node.subgraph_entry in nid or nid == data_node.subgraph_entry
            for nid in wf.nodes
        )
        assert found, (
            f"subgraph_entry={data_node.subgraph_entry!r} "
            f"not found in nodes: {list(wf.nodes.keys())}"
        )
