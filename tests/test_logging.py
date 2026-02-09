"""Tests for logging configuration."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from ai_workers.common.logging import setup_logging


@patch("ai_workers.common.logging.logger")
def test_setup_logging_default(mock_logger: MagicMock) -> None:
    """Test default logging configuration."""
    setup_logging()

    mock_logger.remove.assert_called_once()
    mock_logger.add.assert_called_once()

    args, kwargs = mock_logger.add.call_args
    assert args[0] == sys.stderr
    assert kwargs["level"] == "INFO"
    assert "format" in kwargs
    assert kwargs["format"] is not None
    # When json_format is False, serialize is not passed, or it is False.
    # The implementation does not pass serialize=False explicitly in the else branch,
    # so we check it's not present or False if present.
    assert kwargs.get("serialize") is not True


@patch("ai_workers.common.logging.logger")
def test_setup_logging_custom_level(mock_logger: MagicMock) -> None:
    """Test logging configuration with custom level."""
    setup_logging(level="DEBUG")

    mock_logger.remove.assert_called_once()
    mock_logger.add.assert_called_once()

    _, kwargs = mock_logger.add.call_args
    assert kwargs["level"] == "DEBUG"


@patch("ai_workers.common.logging.logger")
def test_setup_logging_json_format(mock_logger: MagicMock) -> None:
    """Test logging configuration with JSON format."""
    setup_logging(json_format=True)

    mock_logger.remove.assert_called_once()
    mock_logger.add.assert_called_once()

    args, kwargs = mock_logger.add.call_args
    assert args[0] == sys.stderr
    assert kwargs["serialize"] is True
    assert "format" not in kwargs
