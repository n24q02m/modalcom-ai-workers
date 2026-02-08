"""Structured logging configuration using loguru."""

from __future__ import annotations

import sys

from loguru import logger


def setup_logging(*, level: str = "INFO", json_format: bool = False) -> None:
    """Configure loguru for structured logging.

    Args:
        level: Minimum log level.
        json_format: If True, output JSON lines (for production/monitoring).
    """
    logger.remove()

    if json_format:
        logger.add(
            sys.stderr,
            level=level,
            serialize=True,  # JSON output
        )
    else:
        logger.add(
            sys.stderr,
            level=level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                "<level>{message}</level>"
            ),
        )
