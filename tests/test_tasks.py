"""Hermetic tests for PositionTask and the DataNode position-eval workflow.

No live LLM (invoke_agent is stubbed) and no live Stockfish (popen_uci is
mocked / the stockfish resolver is patched). All tests are deterministic.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import chess
import chess.engine
import pytest

from chess_evolve import position_eval
from chess_evolve.pipeline import (
    POSITION_TASK_REF,
    build_position_eval_workflow,
)
from chess_evolve.tasks import PositionTask, load_positions

REPO_ROOT = Path(__file__).resolve().parents[1]
POSITIONS_FILE = REPO_ROOT / "eval" / "test_positions.json"


@pytest.fixture(autouse=True)
def _positions_env(monkeypatch):
    monkeypatch.setenv("CHESS_POSITIONS_FILE", str(POSITIONS_FILE))


# ── Fake Stockfish engine ────────────────────────────────────────


class _FakeEngine:
    """Returns pre-canned analyse() results in call order; ignores quit()."""

    def __init__(self, results: list[dict]):
        self._results = list(results)
        self._i = 0

    def analyse(self, board, limit):  # noqa: ARG002
        result = self._results[self._i]
        self._i += 1
        return result

    def quit(self):  # noqa: D401
        pass


def _cp(value: int, turn: bool) -> chess.engine.PovScore:
    return chess.engine.PovScore(chess.engine.Cp(value), turn)


def _mate(moves: int, turn: bool) -> chess.engine.PovScore:
    return chess.engine.PovScore(chess.engine.Mate(moves), turn)


# ── instances / setup / prompt ───────────────────────────────────


class TestInstances:
    def test_load_positions_valid(self):
        positions = load_positions(POSITIONS_FILE)
        assert len(positions) >= 5
        for p in positions:
            chess.Board(p["fen"])  # must not raise

    def test_instances_yield_fen_metadata(self):
        insts = list(PositionTask().instances())
        assert len(insts) >= 5
        for inst in insts:
            assert inst.id
            assert "fen" in inst.metadata
            chess.Board(inst.metadata["fen"])

    def test_invalid_fen_fails_fast(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps([{"id": "x", "fen": "not-a-fen"}]))
        with pytest.raises(ValueError):
            load_positions(bad)

    def test_missing_fen_fails_fast(self, tmp_path):
        bad = tmp_path / "nofen.json"
        bad.write_text(json.dumps([{"id": "x"}]))
        with pytest.raises(ValueError):
            load_positions(bad)


class TestSetup:
    def test_setup_writes_board_state(self, tmp_path):
        inst = next(iter(PositionTask().instances()))
        PositionTask().setup(inst, tmp_path)
        board_file = tmp_path / ".factory" / "chess" / "board_state.md"
        assert board_file.exists()
        # python-chess normalizes the FEN (e.g. drops an unreachable ep square).
        normalized = chess.Board(inst.metadata["fen"]).fen()
        assert normalized in board_file.read_text()

    def test_setup_creates_memory_file(self, tmp_path):
        """Regression: setup() must create memory.md for DataNode reads."""
        inst = next(iter(PositionTask().instances()))
        PositionTask().setup(inst, tmp_path)
        memory = tmp_path / ".factory" / "chess" / "memory.md"
        assert memory.exists()

    def test_setup_does_not_wipe_workspace(self, tmp_path):
        # A pre-existing sibling file must survive setup (no rmtree).
        keep = tmp_path / "keep.txt"
        keep.write_text("preserve me")
        inst = next(iter(PositionTask().instances()))
        PositionTask().setup(inst, tmp_path)
        assert keep.exists()
        assert keep.read_text() == "preserve me"


class TestPrompt:
    def test_prompt_contains_fen_and_uci_instruction(self):
        inst = next(iter(PositionTask().instances()))
        text = PositionTask().prompt(inst)
        normalized = chess.Board(inst.metadata["fen"]).fen()
        assert normalized in text
        assert "UCI" in text


# ── verify (CPL math), Stockfish mocked ──────────────────────────


class TestVerify:
    def _write_move(self, workspace: Path, move: str) -> None:
        chess_dir = workspace / ".factory" / "chess"
        chess_dir.mkdir(parents=True, exist_ok=True)
        (chess_dir / "move.md").write_text(move)

    def _patch_engine(self, monkeypatch, results):
        fake = _FakeEngine(results)
        monkeypatch.setattr(
            chess.engine.SimpleEngine, "popen_uci",
            lambda *a, **k: fake,
        )
        monkeypatch.setattr(
            "chess_evolve.tasks.resolve_stockfish", lambda: "/fake/stockfish",
        )

    def test_best_move_gives_zero_cpl(self, monkeypatch, tmp_path):
        fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
        board = chess.Board(fen)
        best_move = next(iter(board.legal_moves))
        after = board.copy()
        after.push(best_move)
        # best eval +50 for mover; after eval -50 for opponent -> mover value +50
        self._patch_engine(monkeypatch, [
            {"score": _cp(50, board.turn), "pv": [best_move]},
            {"score": _cp(-50, after.turn)},
        ])
        self._write_move(tmp_path, best_move.uci())
        inst = PositionTask().instances().__next__()
        inst.metadata["fen"] = fen
        result = PositionTask().verify(inst, tmp_path)
        assert result.details["cpl"] == 0
        assert result.score == pytest.approx(1.0)
        assert result.passed is True

    def test_blunder_gives_high_cpl(self, monkeypatch, tmp_path):
        fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
        board = chess.Board(fen)
        moves = list(board.legal_moves)
        best_move, played = moves[0], moves[1]
        after = board.copy()
        after.push(played)
        # best value +50; played leaves opponent +30 -> mover value -30; cpl=80
        self._patch_engine(monkeypatch, [
            {"score": _cp(50, board.turn), "pv": [best_move]},
            {"score": _cp(30, after.turn)},
        ])
        self._write_move(tmp_path, played.uci())
        inst = PositionTask().instances().__next__()
        inst.metadata["fen"] = fen
        result = PositionTask().verify(inst, tmp_path)
        assert result.details["cpl"] == pytest.approx(80.0)
        assert result.score == pytest.approx(0.2)
        assert result.passed is False

    def test_both_mate_scale_cpl_zero(self, monkeypatch, tmp_path):
        fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
        board = chess.Board(fen)
        best_move = next(iter(board.legal_moves))
        after = board.copy()
        after.push(best_move)
        # Both sides on mate scale -> CPL treated as 0.
        self._patch_engine(monkeypatch, [
            {"score": _mate(2, board.turn), "pv": [best_move]},
            {"score": _mate(-3, after.turn)},
        ])
        self._write_move(tmp_path, best_move.uci())
        inst = PositionTask().instances().__next__()
        inst.metadata["fen"] = fen
        result = PositionTask().verify(inst, tmp_path)
        assert result.details["cpl"] == 0.0
        assert result.passed is True

    def test_terminal_position(self, tmp_path):
        # Fool's mate — checkmate, game over, no move required.
        fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        inst = PositionTask().instances().__next__()
        inst.metadata["fen"] = fen
        result = PositionTask().verify(inst, tmp_path)
        assert result.passed is True
        assert result.details.get("note")

    def test_stockfish_not_found_degrades(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "chess_evolve.tasks.resolve_stockfish", lambda: None,
        )
        fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
        inst = PositionTask().instances().__next__()
        inst.metadata["fen"] = fen
        result = PositionTask().verify(inst, tmp_path)
        assert result.passed is False
        assert result.details["error"] == "stockfish_not_found"


# ── DataNode workflow shape + full run (stubbed LLM + Stockfish) ──


class TestWorkflow:
    def test_workflow_uses_task_ref_not_inline(self):
        wf = build_position_eval_workflow()
        data_node = wf.nodes["positions"]
        assert data_node.task_ref == POSITION_TASK_REF
        assert not data_node.inline_items
        assert data_node.source_path is None
        assert "generator" in wf.nodes
        assert data_node.subgraph_entry == "generator"

    def test_datanode_run_hermetic(self, monkeypatch, tmp_path):
        # Stub the LLM: return a fixed legal move for whatever position.
        async def _fake_invoke(role, task, project_path, model=None,
                               timeout=25.0, **kwargs):
            board_file = Path(project_path) / ".factory" / "chess" / "board_state.md"
            fen_line = next(
                line for line in board_file.read_text().splitlines()
                if line.startswith("FEN:")
            )
            fen = fen_line.split("FEN:", 1)[1].strip()
            board = chess.Board(fen)
            move = next(iter(board.legal_moves)).uci()
            move_file = Path(project_path) / ".factory" / "chess" / "move.md"
            move_file.write_text(move)
            return move, 0

        import factory.agents.runner as runner
        monkeypatch.setattr(runner, "invoke_agent", _fake_invoke)

        # Mock Stockfish: every move scored as best (cpl 0) for determinism.
        class _AlwaysBest:
            def analyse(self, board, limit):  # noqa: ARG002
                return {
                    "score": _cp(0, board.turn),
                    "pv": [next(iter(board.legal_moves))],
                }

            def quit(self):
                pass

        monkeypatch.setattr(
            chess.engine.SimpleEngine, "popen_uci", lambda *a, **k: _AlwaysBest(),
        )
        monkeypatch.setattr(
            "chess_evolve.tasks.resolve_stockfish", lambda: "/fake/stockfish",
        )

        aggregate = asyncio.run(position_eval.run_position_eval(
            positions_file=str(POSITIONS_FILE), depth=1, workspace=tmp_path,
        ))

        assert aggregate["count"] >= 5
        assert not aggregate["halted"], aggregate.get("halt_reason")
        assert len(aggregate["per_instance"]) == aggregate["count"]
        for item in aggregate["per_instance"]:
            assert item["move"] is not None
        # Aggregate JSON written under .factory/chess/.
        out = tmp_path / ".factory" / "chess" / "eval_results.json"
        assert out.exists()
        assert json.loads(out.read_text())["count"] == aggregate["count"]

    def test_datanode_run_halts_on_invalid_fen(self, monkeypatch, tmp_path):
        # Invalid FEN must halt the workflow so the aggregate signals halted.
        bad_file = tmp_path / "bad_positions.json"
        bad_file.write_text(json.dumps([
            {"id": "bad-one", "phase": "opening", "fen": "not-a-valid-fen"},
        ]))

        async def _fake_invoke(role, task, project_path, model=None,
                               timeout=25.0, **kwargs):
            return "e2e4", 0

        import factory.agents.runner as runner
        monkeypatch.setattr(runner, "invoke_agent", _fake_invoke)
        monkeypatch.setattr(
            "chess_evolve.tasks.resolve_stockfish", lambda: "/fake/stockfish",
        )

        aggregate = asyncio.run(position_eval.run_position_eval(
            positions_file=str(bad_file), depth=1, workspace=tmp_path,
        ))

        assert aggregate["halted"] is True
        assert aggregate.get("halt_reason")
        assert aggregate["per_instance"] == []
