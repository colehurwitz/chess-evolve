"""CLI entry point for chess-evolve."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import resource
import signal
import subprocess
import sys
import traceback


def _setup_crash_reporting() -> None:
    """Log memory, signals, and unhandled exceptions to stderr."""
    def _signal_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // (1024 * 1024)
        print(
            f"\n[CRASH] Signal {sig_name} ({signum}) received. "
            f"Peak memory: {mb}MB. PID: {os.getpid()}",
            file=sys.stderr, flush=True,
        )
        traceback.print_stack(frame, file=sys.stderr)
        sys.exit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, _signal_handler)

    def _exception_hook(exc_type, exc_value, exc_tb):
        mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // (1024 * 1024)
        print(
            f"\n[CRASH] Unhandled {exc_type.__name__}: {exc_value}. "
            f"Peak memory: {mb}MB",
            file=sys.stderr, flush=True,
        )
        traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.stderr)

    sys.excepthook = _exception_hook


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chess prompt evolution via remote-factory",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("run", help="Run the evolution loop")

    serve_parser = sub.add_parser("serve", help="Start the live web UI")
    serve_parser.add_argument("--port", type=int, default=8422)
    serve_parser.add_argument("--host", type=str, default="0.0.0.0")

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
            asyncio.run(evolve_main())
        except BaseException as exc:
            mb = resource.getrusage(
                resource.RUSAGE_SELF,
            ).ru_maxrss // (1024 * 1024)
            print(
                f"\n[CRASH] {type(exc).__name__}: {exc}. "
                f"Peak memory: {mb}MB",
                file=sys.stderr, flush=True,
            )
            traceback.print_exc(file=sys.stderr)
            raise
    elif args.command == "serve":
        subprocess.run(
            [sys.executable, "-m", "uvicorn", "chess_evolve.serve:app",
             "--host", args.host, "--port", str(args.port),
             "--log-level", "warning"],
        )
    elif args.command == "eval-positions":
        from chess_evolve.position_eval import run_position_eval

        aggregate = asyncio.run(run_position_eval(
            positions_file=args.positions,
            depth=args.depth,
            time_limit=args.time_limit,
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
        # Propagate workflow halts/errors (e.g. invalid FEN) to the shell so
        # CI and downstream `$?` checks see the failure. Successful evals exit 0.
        if aggregate.get("halted"):
            halt_reason = aggregate.get("halt_reason") or "workflow halted"
            print(f"[error] eval halted: {halt_reason}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
