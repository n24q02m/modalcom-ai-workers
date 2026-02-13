"""Tests for Bearer token authentication middleware.

Validates token verification, timing-safe comparison, and dev mode bypass.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from ai_workers.common.auth import verify_api_key


def _make_request(*, auth_header: str | None = None) -> MagicMock:
    """Create a mock FastAPI Request with optional Authorization header."""
    request = MagicMock()
    request.client = MagicMock()
    request.client.__str__ = lambda self: "127.0.0.1:12345"
    headers: dict[str, str] = {}
    if auth_header is not None:
        headers["Authorization"] = auth_header
    request.headers = headers
    return request


class TestVerifyApiKeyDevMode:
    """Test authentication is skipped when WORKER_API_KEY is not set."""

    @pytest.mark.asyncio
    async def test_skip_auth_when_no_key(self) -> None:
        """When WORKER_API_KEY is empty, auth should be skipped."""
        with patch.dict(os.environ, {"WORKER_API_KEY": ""}, clear=False):
            request = _make_request()
            await verify_api_key(request)  # Should not raise

    @pytest.mark.asyncio
    async def test_skip_auth_when_key_unset(self) -> None:
        """When WORKER_API_KEY is not in env at all, auth should be skipped."""
        env = dict(os.environ)
        env.pop("WORKER_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            request = _make_request()
            await verify_api_key(request)  # Should not raise


class TestVerifyApiKeyValidation:
    """Test token validation logic."""

    @pytest.mark.asyncio
    async def test_valid_token(self) -> None:
        """Valid Bearer token should pass."""
        with patch.dict(os.environ, {"WORKER_API_KEY": "test-secret-key"}, clear=False):
            request = _make_request(auth_header="Bearer test-secret-key")
            await verify_api_key(request)  # Should not raise

    @pytest.mark.asyncio
    async def test_valid_token_with_extra_spaces(self) -> None:
        """Token with trailing spaces should be stripped and still valid."""
        with patch.dict(os.environ, {"WORKER_API_KEY": "my-key"}, clear=False):
            request = _make_request(auth_header="Bearer my-key  ")
            await verify_api_key(request)  # Should not raise

    @pytest.mark.asyncio
    async def test_missing_authorization_header(self) -> None:
        """Missing Authorization header should return 401."""
        with patch.dict(os.environ, {"WORKER_API_KEY": "secret"}, clear=False):
            request = _make_request()
            with pytest.raises(HTTPException) as exc_info:
                await verify_api_key(request)
            assert exc_info.value.status_code == 401
            assert "Missing Bearer token" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_empty_authorization_header(self) -> None:
        """Empty Authorization header should return 401."""
        with patch.dict(os.environ, {"WORKER_API_KEY": "secret"}, clear=False):
            request = _make_request(auth_header="")
            with pytest.raises(HTTPException) as exc_info:
                await verify_api_key(request)
            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_non_bearer_scheme(self) -> None:
        """Non-Bearer auth scheme should return 401."""
        with patch.dict(os.environ, {"WORKER_API_KEY": "secret"}, clear=False):
            request = _make_request(auth_header="Basic dXNlcjpwYXNz")
            with pytest.raises(HTTPException) as exc_info:
                await verify_api_key(request)
            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_token(self) -> None:
        """Invalid Bearer token should return 401."""
        with patch.dict(os.environ, {"WORKER_API_KEY": "correct-key"}, clear=False):
            request = _make_request(auth_header="Bearer wrong-key")
            with pytest.raises(HTTPException) as exc_info:
                await verify_api_key(request)
            assert exc_info.value.status_code == 401
            assert "Invalid API key" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_bearer_only_prefix(self) -> None:
        """'Bearer ' with no token should return 401 (empty token != key)."""
        with patch.dict(os.environ, {"WORKER_API_KEY": "secret"}, clear=False):
            request = _make_request(auth_header="Bearer ")
            with pytest.raises(HTTPException) as exc_info:
                await verify_api_key(request)
            assert exc_info.value.status_code == 401


class TestTimingSafety:
    """Test that token comparison uses constant-time comparison."""

    @pytest.mark.asyncio
    async def test_uses_hmac_compare_digest(self) -> None:
        """Verify hmac.compare_digest is used (not direct string comparison)."""
        import hmac

        with (
            patch.dict(os.environ, {"WORKER_API_KEY": "secret"}, clear=False),
            patch.object(hmac, "compare_digest", return_value=True) as mock_compare,
        ):
            request = _make_request(auth_header="Bearer secret")
            await verify_api_key(request)
            mock_compare.assert_called_once()

    @pytest.mark.asyncio
    async def test_timing_safe_rejection(self) -> None:
        """Verify hmac.compare_digest is called even for wrong tokens."""
        import hmac

        with (
            patch.dict(os.environ, {"WORKER_API_KEY": "secret"}, clear=False),
            patch.object(hmac, "compare_digest", return_value=False) as mock_compare,
        ):
            request = _make_request(auth_header="Bearer wrong")
            with pytest.raises(HTTPException):
                await verify_api_key(request)
            mock_compare.assert_called_once()


class TestEdgeCases:
    """Test edge cases and security boundaries."""

    @pytest.mark.asyncio
    async def test_unicode_token(self) -> None:
        """Unicode characters in token should be handled correctly."""
        unicode_key = "key-with-unicode-\u00e9\u00e8\u00ea"
        with patch.dict(os.environ, {"WORKER_API_KEY": unicode_key}, clear=False):
            request = _make_request(auth_header=f"Bearer {unicode_key}")
            await verify_api_key(request)  # Should not raise

    @pytest.mark.asyncio
    async def test_very_long_token(self) -> None:
        """Very long token should not cause issues."""
        long_key = "a" * 10000
        with patch.dict(os.environ, {"WORKER_API_KEY": long_key}, clear=False):
            request = _make_request(auth_header=f"Bearer {long_key}")
            await verify_api_key(request)  # Should not raise

    @pytest.mark.asyncio
    async def test_token_case_sensitive(self) -> None:
        """Token comparison should be case-sensitive."""
        with patch.dict(os.environ, {"WORKER_API_KEY": "MySecret"}, clear=False):
            request = _make_request(auth_header="Bearer mysecret")
            with pytest.raises(HTTPException):
                await verify_api_key(request)
