"""Hermetic tests for ChessSwarmEvaluator.

No live LLM (invoke_agent is stubbed) and no live Stockfish (popen_uci is
mocked). All tests are deterministic.
"""

from __future__ import annotations

from pathlib import Path

import chess
import chess.engine
import pytest
from factory.outer_loop import SwarmConfig
from factory.outer_loop.evaluator import EvalResult

from chess_evolve.evaluator import ChessSwarmEvaluator
from chess_evolve.pipeline import PipelineConfig, build_position_eval_workflow

REPO_ROOT = Path(__file__).resolve().parents[1]
POSITIONS_FILE = REPO_ROOT / "eval" / "test_positions.json"


# ── helpers ──────────────────────────────────────────────────────


def _cp(value: int, turn: bool) -> chess.engine.PovScore:
    return chess.engine.PovScore(chess.engine.Cp(value), turn)


class _AlwaysBestEngine:
    """Stockfish stub: every move evaluated as best (cpl 0)."""

    def analyse(self, board, limit):  # noqa: ARG002
        return {
            "score": _cp(0, board.turn),
            "pv": [next(iter(board.legal_moves))],
        }

    def quit(self):
        pass


def _make_config() -> SwarmConfig:
    return SwarmConfig(
        benchmark="chess-evolve",
        budget=10,
        population_size=4,
        frozen_node_ids=["positions"],
        task_module="chess_evolve.tasks:PositionTask",
    )


# ── fixtures ─────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _positions_env(monkeypatch):
    monkeypatch.setenv("CHESS_POSITIONS_FILE", str(POSITIONS_FILE))


@pytest.fixture()
def _stub_invoke(monkeypatch):
    """Stub invoke_agent: read the board, pick the first legal move."""

    async def _fake_invoke(
        role, task, project_path, model=None, timeout=25.0, **kwargs,
    ):
        board_file = Path(project_path) / ".factory" / "chess" / "board_state.md"
        fen = "startpos"
        if board_file.exists():
            for line in board_file.read_text().splitlines():
                if line.startswith("FEN:"):
                    fen = line.split("FEN:", 1)[1].strip()
                    break
        board = chess.Board(fen)
        move = next(iter(board.legal_moves)).uci()
        move_file = Path(project_path) / ".factory" / "chess" / "move.md"
        move_file.parent.mkdir(parents=True, exist_ok=True)
        move_file.write_text(move)
        return move, 0

    import factory.agents.runner as runner

    monkeypatch.setattr(runner, "invoke_agent", _fake_invoke)


@pytest.fixture()
def _stub_stockfish(monkeypatch):
    """Mock Stockfish: every move evaluated as best (cpl 0)."""
    monkeypatch.setattr(
        chess.engine.SimpleEngine,
        "popen_uci",
        lambda *a, **k: _AlwaysBestEngine(),
    )
    monkeypatch.setattr(
        "chess_evolve.tasks.resolve_stockfish", lambda: "/fake/stockfish",
    )


# ── tests ────────────────────────────────────────────────────────


class TestEvaluatorReturnsEvalResult:
    @pytest.mark.usefixtures("_stub_invoke", "_stub_stockfish")
    def test_evaluate_returns_eval_result(self, tmp_path):
        wf = build_position_eval_workflow(PipelineConfig())
        config = _make_config()
        evaluator = ChessSwarmEvaluator(project_dir=tmp_path, config=config)

        result = evaluator.evaluate(
            wf,
            project_dir=str(tmp_path),
            instances=[],
            individual_id="test-ind-001",
        )

        assert isinstance(result, EvalResult)
        assert result.score >= 0.0
        assert result.benchmark_score >= 0.0


class TestEvaluatorCycleRecord:
    @pytest.mark.usefixtures("_stub_invoke", "_stub_stockfish")
    def test_caches_cycle_record(self, tmp_path):
        wf = build_position_eval_workflow(PipelineConfig())
        config = _make_config()
        evaluator = ChessSwarmEvaluator(project_dir=tmp_path, config=config)

        evaluator.evaluate(
            wf,
            project_dir=str(tmp_path),
            instances=[],
            individual_id="test-ind-002",
        )

        record = evaluator.get_cycle_record("test-ind-002")
        assert record is not None
        assert record.score_end is not None


class TestInvokeAgentRestored:
    def test_invoke_agent_restored_on_exception(self, monkeypatch, tmp_path):
        """Verify the monkey-patch is cleaned up even when InnerLoop.step() raises."""
        import factory.agents.runner as runner

        # Stub invoke_agent to something recognisable
        sentinel = object()
        monkeypatch.setattr(runner, "invoke_agent", sentinel)

        wf = build_position_eval_workflow(PipelineConfig())
        config = _make_config()
        evaluator = ChessSwarmEvaluator(project_dir=tmp_path, config=config)

        # Patch InnerLoop.step to raise
        from factory.inner_loop import InnerLoop

        def _boom(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr(InnerLoop, "step", _boom)

        with pytest.raises(RuntimeError, match="boom"):
            evaluator.evaluate(wf, str(tmp_path), [], individual_id="x")

        # invoke_agent must be restored to the sentinel we set
        assert runner.invoke_agent is sentinel


class TestEvaluatorWithMutatedWorkflow:
    @pytest.mark.usefixtures("_stub_invoke", "_stub_stockfish")
    def test_mutated_workflow_still_evaluates(self, tmp_path):
        """A workflow with different knob_values still produces a valid score."""
        wf = build_position_eval_workflow(PipelineConfig(max_retries=5))
        config = _make_config()
        evaluator = ChessSwarmEvaluator(project_dir=tmp_path, config=config)

        result = evaluator.evaluate(
            wf, str(tmp_path), [], individual_id="mutated-001",
        )

        assert isinstance(result, EvalResult)
        assert result.score >= 0.0


class TestFrozenNodeRejection:
    def test_missing_frozen_node_rejected(self, tmp_path):
        """Workflow missing the frozen 'positions' DataNode → score 0."""
        from factory.workflow.primitives import AgentNode, AgentRole, Workflow

        # Build a workflow WITHOUT the "positions" DataNode
        node = AgentNode(
            id="generator",
            role=AgentRole.STRATEGIST,
            prompt_template="pick a move",
        )
        wf = Workflow(
            name="bad", nodes={"generator": node}, edges=[], start_node="generator",
        )
        config = _make_config()
        evaluator = ChessSwarmEvaluator(project_dir=tmp_path, config=config)

        result = evaluator.evaluate(wf, str(tmp_path), [])
        assert result.score == 0.0
        assert result.details.get("rejected") == "frozen_node_violated"
