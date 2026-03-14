from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException, Request

from ai_workers.common.auth import verify_api_key


@pytest.mark.asyncio
async def test_verify_api_key_valid():
    """Test verification with a valid API key."""
    request = MagicMock(spec=Request)
    request.headers = {"Authorization": "Bearer valid-key"}
    request.client = "test-client"

    with patch("os.getenv", return_value="valid-key"):
        # Should not raise exception
        await verify_api_key(request)


@pytest.mark.asyncio
async def test_verify_api_key_invalid():
    """Test verification with an invalid API key."""
    request = MagicMock(spec=Request)
    request.headers = {"Authorization": "Bearer invalid-key"}
    request.client = "test-client"

    with patch("os.getenv", return_value="valid-key"):
        with pytest.raises(HTTPException) as excinfo:
            await verify_api_key(request)
        assert excinfo.value.status_code == 401
        assert excinfo.value.detail == "Invalid API key"


@pytest.mark.asyncio
async def test_verify_api_key_missing_header():
    """Test verification with missing Authorization header."""
    request = MagicMock(spec=Request)
    request.headers = {}
    request.client = "test-client"

    with patch("os.getenv", return_value="valid-key"):
        with pytest.raises(HTTPException) as excinfo:
            await verify_api_key(request)
        assert excinfo.value.status_code == 401
        assert excinfo.value.detail == "Missing Bearer token"


@pytest.mark.asyncio
async def test_verify_api_key_malformed_header():
    """Test verification with malformed Authorization header."""
    request = MagicMock(spec=Request)
    request.headers = {"Authorization": "Basic whatever"}
    request.client = "test-client"

    with patch("os.getenv", return_value="valid-key"):
        with pytest.raises(HTTPException) as excinfo:
            await verify_api_key(request)
        assert excinfo.value.status_code == 401
        assert excinfo.value.detail == "Missing Bearer token"


@pytest.mark.asyncio
async def test_verify_api_key_dev_mode():
    """Test skipping verification in dev mode (no key configured)."""
    request = MagicMock(spec=Request)
    request.headers = {}  # No header needed
    request.client = "test-client"

    with patch("os.getenv", return_value=""):
        # Should not raise exception
        await verify_api_key(request)
