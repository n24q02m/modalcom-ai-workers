"""Bearer token authentication middleware for FastAPI workers."""

from __future__ import annotations

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from loguru import logger


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


async def auth_middleware(request: Request, call_next):
    """Middleware to enforce authentication on all routes except health checks."""
    if request.url.path in ("/health", "/"):
        return await call_next(request)

    try:
        await verify_api_key(request)
    except HTTPException as exc:
        return JSONResponse(
            status_code=exc.status_code, content={"detail": exc.detail}, headers=exc.headers
        )

    return await call_next(request)
