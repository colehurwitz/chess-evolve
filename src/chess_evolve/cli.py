"""CLI entry point for chess-evolve."""

from __future__ import annotations

import argparse
import logging
import os
import resource
import signal
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
            evolve_main()
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
    else:
        parser.print_help()
