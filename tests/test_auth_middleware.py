import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi import Request

from ai_workers.common.auth import auth_middleware

@pytest.mark.asyncio
async def test_auth_middleware_health_check():
    """Test that health check skips authentication."""
    request = MagicMock(spec=Request)
    request.url.path = "/health"
    call_next = AsyncMock(return_value="response")

    # Mock verify_api_key to ensure it's NOT called
    with patch("ai_workers.common.auth.verify_api_key", new_callable=AsyncMock) as mock_verify:
        response = await auth_middleware(request, call_next)

        mock_verify.assert_not_called()
        call_next.assert_called_once_with(request)
        assert response == "response"

@pytest.mark.asyncio
async def test_auth_middleware_root_check():
    """Test that root path skips authentication."""
    request = MagicMock(spec=Request)
    request.url.path = "/"
    call_next = AsyncMock(return_value="response")

    with patch("ai_workers.common.auth.verify_api_key", new_callable=AsyncMock) as mock_verify:
        response = await auth_middleware(request, call_next)

        mock_verify.assert_not_called()
        call_next.assert_called_once_with(request)
        assert response == "response"

@pytest.mark.asyncio
async def test_auth_middleware_protected_route():
    """Test that other routes trigger authentication."""
    request = MagicMock(spec=Request)
    request.url.path = "/v1/chat/completions"
    call_next = AsyncMock(return_value="response")

    with patch("ai_workers.common.auth.verify_api_key", new_callable=AsyncMock) as mock_verify:
        response = await auth_middleware(request, call_next)

        mock_verify.assert_called_once_with(request)
        call_next.assert_called_once_with(request)
        assert response == "response"
