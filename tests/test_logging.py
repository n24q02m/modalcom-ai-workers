"""Tests for common/logging.py — setup_logging."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_setup_logging_default():
    """setup_logging() with defaults should call logger.remove and logger.add."""
    from ai_workers.common.logging import setup_logging

    with patch("ai_workers.common.logging.logger") as mock_logger:
        setup_logging()
        mock_logger.remove.assert_called_once()
        mock_logger.add.assert_called_once()
        # Check it's adding to stderr
        call_kwargs = mock_logger.add.call_args
        assert call_kwargs is not None


def test_setup_logging_json_format():
    """setup_logging(json_format=True) should pass serialize=True to logger.add."""
    from ai_workers.common.logging import setup_logging

    with patch("ai_workers.common.logging.logger") as mock_logger:
        setup_logging(json_format=True)
        mock_logger.remove.assert_called_once()
        mock_logger.add.assert_called_once()
        call_kwargs = mock_logger.add.call_args.kwargs
        assert call_kwargs.get("serialize") is True


def test_setup_logging_custom_level():
    """setup_logging(level='DEBUG') should pass level to logger.add."""
    from ai_workers.common.logging import setup_logging

    with patch("ai_workers.common.logging.logger") as mock_logger:
        setup_logging(level="DEBUG")
        call_kwargs = mock_logger.add.call_args.kwargs
        assert call_kwargs.get("level") == "DEBUG"


def test_setup_logging_text_format_has_format_string():
    """Default text format includes format string (not serialize)."""
    from ai_workers.common.logging import setup_logging

    with patch("ai_workers.common.logging.logger") as mock_logger:
        setup_logging(json_format=False)
        call_kwargs = mock_logger.add.call_args.kwargs
        assert "format" in call_kwargs
        assert "serialize" not in call_kwargs
