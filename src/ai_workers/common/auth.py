"""Bearer token authentication middleware for FastAPI workers."""

from __future__ import annotations

import hmac
import os

from fastapi import HTTPException, Request, status
from loguru import logger


async def verify_api_key(request: Request) -> None:
    """Verify Bearer token from Authorization header.

    The expected token is resolved from ``API_KEY`` env var, falling back to
    ``WORKER_API_KEY``. If neither is set, auth is skipped entirely (dev mode).
    When a key IS configured, a valid Bearer token is required.

    Uses ``hmac.compare_digest`` to prevent timing attacks.
    """
    expected_key = os.getenv("API_KEY") or os.getenv("WORKER_API_KEY", "")

    # Dev mode: neither API_KEY nor WORKER_API_KEY set → skip auth entirely
    if not expected_key:
        return

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        logger.warning("Missing Bearer token from {}", request.client)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token",
        )

    token = auth_header.removeprefix("Bearer ").strip()

    if not hmac.compare_digest(token.encode(), expected_key.encode()):
        logger.warning("Invalid API key from {}", request.client)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
