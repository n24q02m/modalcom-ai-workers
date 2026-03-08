"""Bearer token authentication middleware for FastAPI workers."""

from __future__ import annotations

import hmac
import os

from fastapi import HTTPException, Request, status
from loguru import logger

# Cache resolved keys at module level (env vars don't change at runtime)
_valid_keys: list[str] | None = None


def _resolve_keys() -> list[str]:
    """Resolve valid API keys from environment variables.

    Supports per-app keys via multiple sources (checked in order):
    1. ``API_KEY`` — single key (backwards compat)
    2. ``WORKER_API_KEY`` — single key (legacy)
    3. ``WORKER_API_KEYS`` — comma-separated multi-key
    4. ``WORKER_API_KEY_*`` — per-app keys (e.g. WORKER_API_KEY_KLPRISM)

    If none are set, returns empty list (dev mode — auth skipped).
    """
    keys: list[str] = []

    # Single key sources
    for var in ("API_KEY", "WORKER_API_KEY"):
        val = os.getenv(var, "").strip()
        if val and val not in keys:
            keys.append(val)

    # Comma-separated multi-key
    multi = os.getenv("WORKER_API_KEYS", "").strip()
    if multi:
        for k in multi.split(","):
            k = k.strip()
            if k and k not in keys:
                keys.append(k)

    # Per-app keys: WORKER_API_KEY_<APP>
    for env_name, env_val in os.environ.items():
        if env_name.startswith("WORKER_API_KEY_") and env_name != "WORKER_API_KEYS":
            val = env_val.strip()
            if val and val not in keys:
                keys.append(val)

    return keys


async def verify_api_key(request: Request) -> None:
    """Verify Bearer token from Authorization header.

    Supports multiple valid keys for per-app isolation.
    If no keys are configured, auth is skipped entirely (dev mode).

    Uses ``hmac.compare_digest`` to prevent timing attacks.
    """
    global _valid_keys
    if _valid_keys is None:
        _valid_keys = _resolve_keys()

    # Dev mode: no keys configured → skip auth entirely
    if not _valid_keys:
        return

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        logger.warning("Missing Bearer token from {}", request.client)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token",
        )

    token = auth_header.removeprefix("Bearer ").strip()
    token_bytes = token.encode()

    if not any(hmac.compare_digest(token_bytes, k.encode()) for k in _valid_keys):
        logger.warning("Invalid API key from {}", request.client)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
