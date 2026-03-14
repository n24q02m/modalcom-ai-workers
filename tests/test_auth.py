"""Tests for authentication middleware."""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Mock dependencies not present in the environment
mock_fastapi = Mock()
mock_fastapi.Request = Mock
mock_fastapi.Response = Mock
mock_fastapi.HTTPException = Exception
mock_fastapi.status = Mock()
mock_fastapi.status.HTTP_401_UNAUTHORIZED = 401
sys.modules["fastapi"] = mock_fastapi

mock_loguru = Mock()
mock_loguru.logger = Mock()
sys.modules["loguru"] = mock_loguru

from ai_workers.common.auth import auth_middleware  # noqa: E402


@pytest.fixture
def mock_verify_api_key():
    with patch("ai_workers.common.auth.verify_api_key", new_callable=AsyncMock) as mock:
        yield mock


def test_auth_middleware_health_check(mock_verify_api_key):
    async def _test():
        request = Mock()
        request.url.path = "/health"
        response_mock = Mock()
        response_mock.status_code = 200
        call_next = AsyncMock(return_value=response_mock)

        response = await auth_middleware(request, call_next)

        assert response.status_code == 200
        mock_verify_api_key.assert_not_called()
        call_next.assert_awaited_once_with(request)

    asyncio.run(_test())


def test_auth_middleware_root(mock_verify_api_key):
    async def _test():
        request = Mock()
        request.url.path = "/"
        response_mock = Mock()
        response_mock.status_code = 200
        call_next = AsyncMock(return_value=response_mock)

        response = await auth_middleware(request, call_next)

        assert response.status_code == 200
        mock_verify_api_key.assert_not_called()
        call_next.assert_awaited_once_with(request)

    asyncio.run(_test())


def test_auth_middleware_protected_route(mock_verify_api_key):
    async def _test():
        request = Mock()
        request.url.path = "/v1/embeddings"
        response_mock = Mock()
        response_mock.status_code = 200
        call_next = AsyncMock(return_value=response_mock)

        response = await auth_middleware(request, call_next)

        assert response.status_code == 200
        mock_verify_api_key.assert_awaited_once_with(request)
        call_next.assert_awaited_once_with(request)

    asyncio.run(_test())
