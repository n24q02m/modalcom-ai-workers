import os
import pytest
from unittest.mock import MagicMock, patch
from fastapi import Request, HTTPException, status

# Import the function to test
from ai_workers.common.auth import verify_api_key

@pytest.mark.asyncio
async def test_auth_failure_when_env_var_missing():
    """Ensure that authentication fails closed when WORKER_API_KEY is not set."""
    # Mock request
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {}

    # Ensure WORKER_API_KEY is unset
    with patch.dict(os.environ, {}, clear=True):
        # This should raise HTTPException with 500
        with pytest.raises(HTTPException) as excinfo:
            await verify_api_key(mock_request)

        assert excinfo.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Server misconfiguration" in excinfo.value.detail

@pytest.mark.asyncio
async def test_auth_enforced_when_env_var_set():
    """Ensure that authentication is enforced when WORKER_API_KEY is set."""
    # Mock request with no auth header
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {}
    mock_request.client = "test_client"

    # Set WORKER_API_KEY
    with patch.dict(os.environ, {"WORKER_API_KEY": "secret-key"}):
        # This SHOULD raise HTTPException with 401 because token is missing
        with pytest.raises(HTTPException) as excinfo:
            await verify_api_key(mock_request)
        assert excinfo.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Missing Bearer token" in excinfo.value.detail

@pytest.mark.asyncio
async def test_auth_success_with_valid_token():
    """Ensure that authentication succeeds with valid token."""
    # Mock request with correct auth header
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {"Authorization": "Bearer secret-key"}
    mock_request.client = "test_client"

    # Set WORKER_API_KEY
    with patch.dict(os.environ, {"WORKER_API_KEY": "secret-key"}):
        # Should NOT raise exception
        await verify_api_key(mock_request)
