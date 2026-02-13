"""Bearer token authentication middleware for FastAPI workers."""

from __future__ import annotations

import hmac
import os

from fastapi import HTTPException, Request, status
from loguru import logger


async def verify_api_key(request: Request) -> None:
    """Verify Bearer token from Authorization header.

    The expected token is set via Modal Secret "worker-api-key".
    If WORKER_API_KEY env var is empty, authentication is skipped (dev mode).

    Uses ``hmac.compare_digest`` to prevent timing attacks.
    """
    expected_key = os.getenv("WORKER_API_KEY", "")

    # Skip auth in dev mode (no key configured)
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
