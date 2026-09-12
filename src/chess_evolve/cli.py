"""CLI entry point for chess-evolve."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import resource
import signal
import sys
import traceback

import structlog

from chess_evolve.logging_config import setup_logging

_crash_log = structlog.get_logger("crash")


def _setup_crash_reporting() -> None:
    """Log memory, signals, and unhandled exceptions via structlog."""
    def _signal_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // (1024 * 1024)
        _crash_log.critical(
            "signal_received", signal=sig_name, signum=signum,
            peak_memory_mb=mb, pid=os.getpid(),
        )
        traceback.print_stack(frame, file=sys.stderr)
        sys.exit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, _signal_handler)

    def _exception_hook(exc_type, exc_value, exc_tb):
        mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // (1024 * 1024)
        _crash_log.critical(
            "unhandled_exception", exc_type=exc_type.__name__,
            exc_value=str(exc_value), peak_memory_mb=mb,
        )
        traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.stderr)

    sys.excepthook = _exception_hook


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Chess prompt evolution via remote-factory",
    )
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Run the evolution loop")
    run_parser.add_argument(
        "--task", choices=["position", "game"], default="position",
        help="Evaluation task: 'position' for static CPL, 'game' for full games",
    )

    serve_parser = sub.add_parser("serve", help="Start the live dashboard")
    serve_parser.add_argument("--port", type=int, default=8422)
    serve_parser.add_argument("--host", type=str, default="0.0.0.0")
    serve_parser.add_argument(
        "--project", type=str, default=None,
        help="Project root directory (default: current directory)",
    )
    serve_parser.add_argument(
        "--replay", type=str, default=None,
        help="Replay a JSONL file instead of live tailing",
    )

    eval_parser = sub.add_parser(
        "eval-positions",
        help="Evaluate move quality on a set of positions (centipawn loss)",
    )
    eval_parser.add_argument(
        "--positions", type=str, default="eval/test_positions.json",
        help="Path to the positions JSON file",
    )
    eval_parser.add_argument(
        "--depth", type=int, default=None,
        help="Stockfish analysis depth (default 12 if neither depth nor time set)",
    )
    eval_parser.add_argument(
        "--time", type=float, default=None, dest="time_limit",
        help="Stockfish analysis time per position in seconds",
    )
    eval_parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Skip LLM calls; verify positions without generating moves",
    )

    args = parser.parse_args()

    if args.command == "run":
        _setup_crash_reporting()
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s %(levelname)s %(message)s",
            stream=sys.stderr,
        )
        try:
            from chess_evolve.evolution import main as evolve_main
            evolve_main(task_type=args.task)
        except BaseException as exc:
            mb = resource.getrusage(
                resource.RUSAGE_SELF,
            ).ru_maxrss // (1024 * 1024)
            _crash_log.critical(
                "evolution_crashed", exc_type=type(exc).__name__,
                exc_value=str(exc), peak_memory_mb=mb,
            )
            traceback.print_exc(file=sys.stderr)
            raise
    elif args.command == "serve":
        try:
            import uvicorn  # noqa: F401

            from chess_evolve.dashboard import create_app
        except ImportError:
            print(
                "Dashboard dependencies not installed. Run:\n"
                "  pip install 'chess-evolve[dashboard]'",
                file=sys.stderr,
            )
            sys.exit(1)

        from pathlib import Path

        project_root = Path(args.project) if args.project else Path.cwd()
        replay_path = Path(args.replay) if args.replay else None
        app = create_app(
            project_root=project_root,
            replay_path=replay_path,
        )
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="warning",
        )
    elif args.command == "eval-positions":
        from chess_evolve.position_eval import run_position_eval

        aggregate = asyncio.run(run_position_eval(
            positions_file=args.positions,
            depth=args.depth,
            time_limit=args.time_limit,
            dry_run=args.dry_run,
        ))
        for item in aggregate["per_instance"]:
            cpl = item.get("cpl")
            cpl_str = f"{cpl:.0f}" if isinstance(cpl, (int, float)) else "n/a"
            print(
                f"{item['id']}: move={item.get('move')} "
                f"best={item.get('best_move')} cpl={cpl_str} "
                f"score={item.get('score', 0.0):.3f} pass={item.get('passed')}"
            )
        mean_cpl = aggregate.get("mean_cpl")
        mean_cpl_str = (
            f"{mean_cpl:.1f}" if isinstance(mean_cpl, (int, float)) else "n/a"
        )
        print(
            f"\nmean_cpl={mean_cpl_str} "
            f"mean_score={aggregate.get('mean_score', 0.0):.3f} "
            f"pass_rate={aggregate.get('pass_rate', 0.0):.2f} "
            f"({aggregate.get('passes', 0)}/{aggregate.get('count', 0)})"
        )
        print(f"results written to {aggregate.get('results_path')}")
        if aggregate.get("halted"):
            halt_reason = aggregate.get("halt_reason") or "workflow halted"
            print(f"[error] eval halted: {halt_reason}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
