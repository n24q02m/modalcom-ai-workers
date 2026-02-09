"""Tests for authentication middleware."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import Request

# This import assumes auth_middleware is added to ai_workers.common.auth
from ai_workers.common.auth import auth_middleware

@pytest.mark.asyncio
async def test_auth_middleware_bypass_health():
    """Test that /health endpoint bypasses authentication."""
    request = MagicMock(spec=Request)
    request.url.path = "/health"
    call_next = AsyncMock(return_value="response")

    # We mock verify_api_key to ensure it's NOT called
    with patch("ai_workers.common.auth.verify_api_key", new_callable=AsyncMock) as mock_verify:
        response = await auth_middleware(request, call_next)

    assert response == "response"
    call_next.assert_called_once_with(request)
    mock_verify.assert_not_called()

@pytest.mark.asyncio
async def test_auth_middleware_bypass_root():
    """Test that / endpoint bypasses authentication."""
    request = MagicMock(spec=Request)
    request.url.path = "/"
    call_next = AsyncMock(return_value="response")

    with patch("ai_workers.common.auth.verify_api_key", new_callable=AsyncMock) as mock_verify:
        response = await auth_middleware(request, call_next)

    assert response == "response"
    call_next.assert_called_once_with(request)
    mock_verify.assert_not_called()

@pytest.mark.asyncio
async def test_auth_middleware_enforce_auth():
    """Test that other endpoints enforce authentication."""
    request = MagicMock(spec=Request)
    request.url.path = "/v1/embeddings"
    call_next = AsyncMock(return_value="response")

    with patch("ai_workers.common.auth.verify_api_key", new_callable=AsyncMock) as mock_verify:
        response = await auth_middleware(request, call_next)

    assert response == "response"
    call_next.assert_called_once_with(request)
    mock_verify.assert_called_once_with(request)
