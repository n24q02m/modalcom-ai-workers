"""Bearer token authentication middleware for FastAPI workers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import HTTPException, Request, Response, status
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


async def verify_api_key(request: Request) -> None:
    """Verify Bearer token from Authorization header.

    The expected token is set via Modal Secret "worker-api-key".
    If WORKER_API_KEY env var is empty, authentication is skipped (dev mode).
    """
    import os

    expected_key = os.getenv("WORKER_API_KEY", "")

    # Skip auth in dev mode (no key configured)
    if not expected_key:
        return

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        logger.warning(f"Missing Bearer token from {request.client}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token",
        )

    token = auth_header.removeprefix("Bearer ").strip()
    if token != expected_key:
        logger.warning(f"Invalid API key from {request.client}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )


async def auth_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Middleware to verify API key for all requests except health checks."""
    if request.url.path in ("/health", "/"):
        return await call_next(request)

    await verify_api_key(request)
    return await call_next(request)
