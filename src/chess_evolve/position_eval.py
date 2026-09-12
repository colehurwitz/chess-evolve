"""Standalone position evaluation — run PositionTask outside SwarmEngine.

Provides ``run_position_eval()``, the async entry-point used by the
``eval-positions`` CLI command.  Each position is set up, then verified
by ``PositionTask.verify()`` (Stockfish CPL scoring).

Note: without an LLM-generated ``move.md``, ``verify()`` will report
"no move parsed" for every position.  Wire a generator (e.g. via
``WorkflowExecutor``) to produce real moves before calling ``verify()``.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from chess_evolve.tasks import PositionTask


async def run_position_eval(
    positions_file: str | None = None,
    depth: int | None = None,
    time_limit: float | None = None,
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

    for inst in instances:
        workspace = Path(tempfile.mkdtemp(prefix=f"chess_eval_{inst.id}_"))
        task.setup(inst, workspace)

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
        "halted": False,
        "halt_reason": None,
    }
