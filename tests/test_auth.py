"""Tests for authentication middleware."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi import Request, HTTPException, status
from ai_workers.common.auth import verify_api_key, auth_middleware

# Mock Request object
def create_mock_request(path="/", headers=None):
    request = MagicMock(spec=Request)
    request.url.path = path
    request.headers = headers or {}
    request.client = "test_client"
    return request

@pytest.mark.asyncio
async def test_verify_api_key_dev_mode():
    """Test verify_api_key when WORKER_API_KEY is not set (dev mode)."""
    with patch("os.getenv", return_value=""):
        request = create_mock_request()
        # Should not raise exception
        await verify_api_key(request)

@pytest.mark.asyncio
async def test_verify_api_key_success():
    """Test verify_api_key with correct key."""
    with patch("os.getenv", return_value="secret_key"):
        request = create_mock_request(headers={"Authorization": "Bearer secret_key"})
        await verify_api_key(request)

@pytest.mark.asyncio
async def test_verify_api_key_missing_header():
    """Test verify_api_key with missing Authorization header."""
    with patch("os.getenv", return_value="secret_key"):
        request = create_mock_request(headers={})
        with pytest.raises(HTTPException) as excinfo:
            await verify_api_key(request)
        assert excinfo.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert excinfo.value.detail == "Missing Bearer token"

@pytest.mark.asyncio
async def test_verify_api_key_invalid_format():
    """Test verify_api_key with invalid Authorization header format."""
    with patch("os.getenv", return_value="secret_key"):
        request = create_mock_request(headers={"Authorization": "Basic user:pass"})
        with pytest.raises(HTTPException) as excinfo:
            await verify_api_key(request)
        assert excinfo.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert excinfo.value.detail == "Missing Bearer token"

@pytest.mark.asyncio
async def test_verify_api_key_invalid_key():
    """Test verify_api_key with wrong key."""
    with patch("os.getenv", return_value="secret_key"):
        request = create_mock_request(headers={"Authorization": "Bearer wrong_key"})
        with pytest.raises(HTTPException) as excinfo:
            await verify_api_key(request)
        assert excinfo.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert excinfo.value.detail == "Invalid API key"

@pytest.mark.asyncio
async def test_auth_middleware_skip_health():
    """Test auth_middleware skips auth for health check."""
    request = create_mock_request(path="/health")
    call_next = AsyncMock()

    # We patch verify_api_key to ensure it's NOT called
    with patch("ai_workers.common.auth.verify_api_key") as mock_verify:
        await auth_middleware(request, call_next)
        mock_verify.assert_not_called()
        call_next.assert_called_once_with(request)

@pytest.mark.asyncio
async def test_auth_middleware_skip_root():
    """Test auth_middleware skips auth for root path."""
    request = create_mock_request(path="/")
    call_next = AsyncMock()

    with patch("ai_workers.common.auth.verify_api_key") as mock_verify:
        await auth_middleware(request, call_next)
        mock_verify.assert_not_called()
        call_next.assert_called_once_with(request)

@pytest.mark.asyncio
async def test_auth_middleware_enforce_auth():
    """Test auth_middleware enforces auth for other paths."""
    request = create_mock_request(path="/v1/rerank")
    call_next = AsyncMock()

    with patch("ai_workers.common.auth.verify_api_key") as mock_verify:
        await auth_middleware(request, call_next)
        mock_verify.assert_called_once_with(request)
        call_next.assert_called_once_with(request)
