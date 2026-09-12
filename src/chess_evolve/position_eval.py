"""Standalone position evaluation — run PositionTask outside SwarmEngine.

Provides ``run_position_eval()``, the async entry-point used by the
``eval-positions`` CLI command.  Each position is set up, the LLM pipeline
is executed via ``WorkflowExecutor`` to generate a move, and then
``PositionTask.verify()`` scores the move against Stockfish (CPL).

When ``dry_run=True``, the LLM pipeline is skipped and ``verify()`` runs
directly (useful for testing without API costs).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import structlog
from factory.workflow.executor import WorkflowExecutor

from chess_evolve.pipeline import build_position_eval_workflow
from chess_evolve.tasks import PositionTask

log = structlog.get_logger()


async def _run_pipeline_for_position(workspace: Path) -> bool:
    """Execute the position-eval workflow to generate a move.

    Uses the same ``_sdk_invoke_agent`` swap pattern as ``engine.py`` to
    route ``AgentNode`` calls through the Claude CLI/API.

    Returns True if execution succeeded, False otherwise.
    """
    from chess_evolve.engine import _sdk_invoke_agent  # noqa: PLC0415

    wf = build_position_eval_workflow()
    # Remove the DataNode wrapper — we only need the generator subgraph
    # since we're driving instances ourselves.
    generator_node = wf.nodes["generator"]
    from factory.workflow.primitives import Workflow  # noqa: PLC0415

    single_node_wf = Workflow(
        name="position-eval-single",
        nodes={"generator": generator_node},
        edges=[],
        start_node="generator",
    )

    executor = WorkflowExecutor(
        workflow=single_node_wf,
        project_path=workspace,
        auto_approve=True,
        agent_fn=_sdk_invoke_agent,
    )
    # Mark board_state.md and memory.md as already available so the
    # executor doesn't block waiting for them.
    executor.completed_files.add(".factory/chess/board_state.md")
    executor.completed_files.add(".factory/chess/memory.md")

    result = await executor.execute()

    # Write outputs to the paths declared in node.writes
    for nid, output in result.node_outputs.items():
        node = single_node_wf.nodes.get(nid)
        if node and node.writes and output:
            for wpath in node.writes:
                fpath = workspace / wpath
                fpath.parent.mkdir(parents=True, exist_ok=True)
                fpath.write_text(output)
                log.debug(
                    "position_eval_write", node_id=nid,
                    path=wpath, chars=len(output),
                )

    if result.halted:
        log.warning("position_eval_halted", reason=result.halt_reason)
        return False
    return True


async def run_position_eval(
    positions_file: str | None = None,
    depth: int | None = None,
    time_limit: float | None = None,
    dry_run: bool = False,
) -> dict:
    """Evaluate positions and return an aggregate results dict.

    Parameters
    ----------
    positions_file:
        Path to the JSON file containing positions.
    depth:
        Stockfish analysis depth (overrides ``CHESS_EVAL_DEPTH`` env).
    time_limit:
        Stockfish analysis time in seconds (overrides ``CHESS_EVAL_TIME`` env).
    dry_run:
        If True, skip the LLM pipeline and run ``verify()`` directly
        (produces "no move parsed" for every position — useful for
        testing without API costs).

    Returns
    -------
    dict with keys: per_instance, mean_cpl, mean_score, pass_rate,
    passes, count, results_path, halted, halt_reason.
    """
    if depth is not None:
        os.environ["CHESS_EVAL_DEPTH"] = str(depth)
    if time_limit is not None:
        os.environ["CHESS_EVAL_TIME"] = str(time_limit)

    task = PositionTask()
    instances = list(task.instances())

    results: list[dict] = []
    passes = 0
    total_cpl = 0.0
    total_score = 0.0
    cpl_count = 0
    any_halted = False
    halt_reason = None

    for inst in instances:
        workspace = Path(tempfile.mkdtemp(prefix=f"chess_eval_{inst.id}_"))
        task.setup(inst, workspace)

        if not dry_run:
            ok = await _run_pipeline_for_position(workspace)
            if not ok:
                any_halted = True
                halt_reason = "pipeline execution halted"
                log.warning("position_eval_instance_halted", instance_id=inst.id)

        result = task.verify(inst, workspace)
        details = result.details or {}

        entry = {
            "id": inst.id,
            "move": details.get("move"),
            "best_move": details.get("best_move"),
            "cpl": details.get("cpl"),
            "score": result.score,
            "passed": result.passed,
            "fen": details.get("fen"),
            "phase": details.get("phase"),
        }
        results.append(entry)

        log.info(
            "position_eval_result", instance_id=inst.id,
            move=details.get("move"), best_move=details.get("best_move"),
            cpl=details.get("cpl"), score=result.score, passed=result.passed,
        )

        if result.passed:
            passes += 1
        total_score += result.score
        cpl = details.get("cpl")
        if isinstance(cpl, (int, float)):
            total_cpl += cpl
            cpl_count += 1

    count = len(results)
    results_path = Path("eval/eval_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))

    return {
        "per_instance": results,
        "mean_cpl": total_cpl / cpl_count if cpl_count > 0 else None,
        "mean_score": total_score / max(count, 1),
        "pass_rate": passes / max(count, 1),
        "passes": passes,
        "count": count,
        "results_path": str(results_path),
        "halted": any_halted,
        "halt_reason": halt_reason,
    }
