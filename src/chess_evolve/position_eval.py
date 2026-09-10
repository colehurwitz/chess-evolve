"""Run the DataNode-driven position evaluation and aggregate the results."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from factory.workflow.executor import WorkflowExecutor

from chess_evolve.config import WORKSPACE
from chess_evolve.pipeline import PipelineConfig, build_position_eval_workflow

DATA_NODE_ID = "positions"


async def _position_move_invoke(
    role: str,
    task: str,
    project_path: Path,
    model: str | None = None,
    timeout: float = 25.0,
    **kwargs: object,
) -> tuple[str, int]:
    """invoke_agent shim for position eval that also persists the move.

    The stock WorkflowExecutor keeps AgentNode output only in memory; the
    PositionTask.verify hook reads ``.factory/chess/move.md`` from disk, so this
    wrapper writes the generated move there after invoking the model.
    """
    from chess_evolve.engine import _sdk_invoke_agent

    text, code = await _sdk_invoke_agent(
        role, task, project_path, model=model, timeout=timeout, **kwargs,
    )
    move_file = Path(project_path) / ".factory" / "chess" / "move.md"
    move_file.parent.mkdir(parents=True, exist_ok=True)
    move_file.write_text(text or "")
    return text, code


def _prime_workspace(workspace: Path) -> None:
    """Create the chess dir and placeholder read-files the generator expects.

    The generator AgentNode declares reads on ``board_state.md`` /
    ``memory.md``; the DataNode only marks reads that already exist on disk as
    satisfied, so we pre-create them (setup() overwrites board_state.md per
    position with the real content).
    """
    chess_dir = workspace / ".factory" / "chess"
    chess_dir.mkdir(parents=True, exist_ok=True)
    for name in ("board_state.md", "memory.md"):
        f = chess_dir / name
        if not f.exists():
            f.write_text("")
    # Clear any stale move from a previous run.
    move_file = chess_dir / "move.md"
    if move_file.exists():
        move_file.unlink()


def _aggregate(item_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the per-position + mean aggregate from raw DataNode item results."""
    per_instance: list[dict[str, Any]] = []
    cpls: list[float] = []
    scores: list[float] = []
    passes = 0
    for item in item_results:
        details = item.get("verify_details", {}) or {}
        cpl = details.get("cpl")
        score = float(item.get("score", 0.0))
        passed = bool(item.get("passed", False))
        scores.append(score)
        if isinstance(cpl, (int, float)):
            cpls.append(float(cpl))
        if passed:
            passes += 1
        per_instance.append({
            "id": item.get("item_id"),
            "move": details.get("move"),
            "best_move": details.get("best_move"),
            "cpl": cpl,
            "score": score,
            "passed": passed,
            "phase": details.get("phase", ""),
            "error": item.get("error") or details.get("error"),
        })

    n = max(len(per_instance), 1)
    return {
        "per_instance": per_instance,
        "count": len(per_instance),
        "mean_cpl": (sum(cpls) / len(cpls)) if cpls else None,
        "mean_score": sum(scores) / n,
        "pass_rate": passes / n,
        "passes": passes,
    }


async def run_position_eval(
    positions_file: str | Path = "eval/test_positions.json",
    depth: int | None = None,
    time_limit: float | None = None,
    workspace: Path | None = None,
    cfg: PipelineConfig | None = None,
) -> dict[str, Any]:
    """Evaluate every position via the DataNode workflow; return the aggregate.

    Writes the aggregate to ``<workspace>/.factory/chess/eval_results.json``.
    """
    os.environ["CHESS_POSITIONS_FILE"] = str(positions_file)
    if time_limit is not None:
        os.environ["CHESS_EVAL_TIME"] = str(time_limit)
        os.environ.pop("CHESS_EVAL_DEPTH", None)
    elif depth is not None:
        os.environ["CHESS_EVAL_DEPTH"] = str(depth)
        os.environ.pop("CHESS_EVAL_TIME", None)

    workspace = workspace or (WORKSPACE / "position-eval")
    workspace = Path(workspace)
    _prime_workspace(workspace)

    wf = build_position_eval_workflow(cfg)

    # The stock executor keeps AgentNode output in memory only; swap invoke_agent
    # for a shim that also writes .factory/chess/move.md so verify() can read it.
    import factory.agents.runner as _runner

    _orig_invoke = _runner.invoke_agent
    _runner.invoke_agent = _position_move_invoke  # type: ignore[assignment]
    try:
        executor = WorkflowExecutor(
            workflow=wf, project_path=workspace, auto_approve=True,
        )
        result = await executor.execute()
    finally:
        _runner.invoke_agent = _orig_invoke  # type: ignore[assignment]

    raw = result.node_outputs.get(DATA_NODE_ID, "[]")
    try:
        item_results = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        item_results = []

    aggregate = _aggregate(item_results)
    aggregate["halted"] = result.halted
    aggregate["halt_reason"] = result.halt_reason

    out_path = workspace / ".factory" / "chess" / "eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(aggregate, indent=2))
    aggregate["results_path"] = str(out_path)
    return aggregate
