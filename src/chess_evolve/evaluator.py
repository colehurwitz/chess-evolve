"""ChessSwarmEvaluator — bridge from SwarmEngine to InnerLoop + DataNode.

SwarmEngine calls evaluate() for each candidate workflow. This adapter:
1. Primes the workspace (board_state.md, memory.md placeholders)
2. Installs the invoke_agent wrapper (writes move.md to disk)
3. Creates InnerLoop with the candidate workflow
4. Calls loop.step() → DataNode auto-detects → _step_with_data_node()
5. Returns EvalResult with aggregated score from PositionTask.verify()
"""

from __future__ import annotations

from pathlib import Path

import factory.agents.runner as _runner
from factory.inner_loop import InnerLoop
from factory.outer_loop import SwarmConfig
from factory.outer_loop.evaluator import EvalResult, SwarmEvaluator
from factory.workflow.primitives import Workflow

from chess_evolve.position_eval import (
    _FACTORY_DEFAULT_INVOKE,
    _make_move_invoke,
    _prime_workspace,
)


def _get_live_or_stub_invoke():
    """Return the inner invoke_agent to wrap.

    If ``_runner.invoke_agent`` has been replaced (e.g. by a test monkeypatch),
    use that replacement.  Otherwise import the live SDK implementation from
    ``chess_evolve.engine``.
    """
    current = _runner.invoke_agent
    if current is not _FACTORY_DEFAULT_INVOKE:
        return current
    from chess_evolve.engine import _sdk_invoke_agent

    return _sdk_invoke_agent


class ChessSwarmEvaluator(SwarmEvaluator):
    """Evaluate a candidate workflow via InnerLoop + DataNode + PositionTask.

    Overrides :meth:`evaluate` to inject the invoke_agent monkey-patch and
    prime the workspace before delegating to InnerLoop.step(), which
    auto-detects the DataNode and routes through _step_with_data_node().
    """

    def __init__(self, project_dir: Path, config: SwarmConfig) -> None:
        super().__init__(config=config, project_dir=project_dir)
        self._project_dir = Path(project_dir)

    def evaluate(
        self,
        workflow: Workflow,
        project_dir: str,
        instances: list[str],
        individual_id: str | None = None,
    ) -> EvalResult:
        """Evaluate *workflow* on chess positions via DataNode + PositionTask."""
        # Cache check (reuse parent infrastructure)
        cached = self._cache.get(workflow, instances)
        if cached is not None:
            score, cost, _ = cached
            return EvalResult(score=score, cost_usd=cost, benchmark_score=score)

        # Frozen-node guard
        if not self._check_frozen_nodes(workflow):
            return EvalResult(
                score=0.0, details={"rejected": "frozen_node_violated"},
            )

        workspace = Path(project_dir) if project_dir else self._project_dir
        _prime_workspace(workspace)

        # Install invoke_agent wrapper so move.md is written to disk
        _orig = _runner.invoke_agent
        inner = _get_live_or_stub_invoke()
        _runner.invoke_agent = _make_move_invoke(inner)  # type: ignore[assignment]
        try:
            from chess_evolve.tasks import PositionTask

            loop = InnerLoop(
                project_dir=workspace,
                workflow=workflow,
                task=PositionTask(),
            )
            record = loop.step()
        finally:
            _runner.invoke_agent = _orig  # type: ignore[assignment]

        score = record.score_end or 0.0
        cost = record.total_cost_usd

        if individual_id:
            self._cycle_records[individual_id] = record

        self._cache.put(workflow, instances, score, cost)

        return EvalResult(
            score=score,
            cost_usd=cost,
            benchmark_score=score,
            details={
                "instance_results": record.instance_results or [],
            },
        )
