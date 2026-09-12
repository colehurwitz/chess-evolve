"""Tests for structured logging configuration."""

from __future__ import annotations

import io
import json

import structlog

from chess_evolve.logging_config import setup_logging


class TestLoggingConfig:
    def test_json_format_contains_expected_fields(self, monkeypatch):
        """JSON log output should contain level, event, and timestamp."""
        monkeypatch.setenv("CHESS_LOG_FORMAT", "json")
        # Reset structlog so setup_logging() reconfigures it
        structlog.reset_defaults()

        buf = io.StringIO()
        monkeypatch.setattr("sys.stderr", buf)

        setup_logging()

        logger = structlog.get_logger()
        logger.info("test_event", game_tag="g1", node_id="generator")

        output = buf.getvalue().strip()
        assert output, "Expected log output on stderr"
        parsed = json.loads(output)
        assert parsed["event"] == "test_event"
        assert parsed["game_tag"] == "g1"
        assert parsed["node_id"] == "generator"
        assert "timestamp" in parsed
        assert "level" in parsed

    def test_console_format_does_not_crash(self, monkeypatch):
        """Console renderer (default) should produce non-empty output."""
        monkeypatch.setenv("CHESS_LOG_FORMAT", "console")
        structlog.reset_defaults()

        buf = io.StringIO()
        monkeypatch.setattr("sys.stderr", buf)

        setup_logging()

        logger = structlog.get_logger()
        logger.info("console_test", game_tag="g2")

        output = buf.getvalue()
        assert "console_test" in output
        assert "g2" in output

    def test_default_is_console(self, monkeypatch):
        """When CHESS_LOG_FORMAT is not set, console renderer is used."""
        monkeypatch.delenv("CHESS_LOG_FORMAT", raising=False)
        structlog.reset_defaults()

        buf = io.StringIO()
        monkeypatch.setattr("sys.stderr", buf)

        setup_logging()

        logger = structlog.get_logger()
        logger.info("default_test")

        output = buf.getvalue()
        assert "default_test" in output
        # Console format should NOT be valid JSON
        try:
            json.loads(output.strip())
            is_json = True
        except (json.JSONDecodeError, ValueError):
            is_json = False
        assert not is_json
