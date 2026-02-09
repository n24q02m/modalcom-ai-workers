"""Bearer token authentication middleware for FastAPI workers."""

from __future__ import annotations

from fastapi import HTTPException, Request, status
from loguru import logger


async def verify_api_key(request: Request) -> None:
    """Verify Bearer token from Authorization header.

    The expected token is set via Modal Secret "worker-api-key".
    Authentication is MANDATORY. If WORKER_API_KEY is not set, the server will
    reject all requests with 500 Internal Server Error to prevent insecure access.
    """
    import os

    expected_key = os.getenv("WORKER_API_KEY", "")

    # Fail securely if no key is configured
    if not expected_key:
        logger.error("WORKER_API_KEY is not set. Authentication cannot be verified.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server misconfiguration: Missing API key",
        )

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
