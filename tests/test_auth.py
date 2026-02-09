import sys
from unittest.mock import MagicMock, patch

import pytest


# Define mocks to be used in tests
class MockHTTPError(Exception):
    def __init__(self, status_code, detail):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)

class MockStatus:
    HTTP_401_UNAUTHORIZED = 401

class MockRequest:
    pass

@pytest.fixture
def auth_module():
    """
    Fixture that mocks fastapi and imports ai_workers.common.auth
    in an isolated environment. This ensures tests pass even if
    fastapi is not installed, and doesn't pollute global state.
    """
    with patch.dict(sys.modules):
        # Create mock fastapi
        mock_fastapi = MagicMock()
        mock_fastapi.HTTPException = MockHTTPError
        mock_fastapi.status = MockStatus()
        mock_fastapi.Request = MockRequest
        sys.modules["fastapi"] = mock_fastapi

        # Remove the module if it exists to force re-import with mocked fastapi
        if "ai_workers.common.auth" in sys.modules:
            del sys.modules["ai_workers.common.auth"]

        import ai_workers.common.auth
        yield ai_workers.common.auth

@pytest.mark.asyncio
async def test_verify_api_key_dev_mode(auth_module, monkeypatch):
    # Ensure WORKER_API_KEY is unset
    monkeypatch.delenv("WORKER_API_KEY", raising=False)
    request = MagicMock()

    # Should not raise exception and return None
    result = await auth_module.verify_api_key(request)
    assert result is None

@pytest.mark.asyncio
async def test_verify_api_key_missing_header(auth_module, monkeypatch):
    monkeypatch.setenv("WORKER_API_KEY", "secret-key")
    request = MagicMock()
    request.headers.get.return_value = ""
    request.client = "test-client"

    with pytest.raises(MockHTTPError) as excinfo:
        await auth_module.verify_api_key(request)

    assert excinfo.value.status_code == 401
    assert excinfo.value.detail == "Missing Bearer token"

@pytest.mark.asyncio
async def test_verify_api_key_invalid_header_scheme(auth_module, monkeypatch):
    monkeypatch.setenv("WORKER_API_KEY", "secret-key")
    request = MagicMock()
    # Header exists but not Bearer
    request.headers.get.return_value = "Basic 12345"
    request.client = "test-client"

    with pytest.raises(MockHTTPError) as excinfo:
        await auth_module.verify_api_key(request)

    assert excinfo.value.status_code == 401
    assert excinfo.value.detail == "Missing Bearer token"

@pytest.mark.asyncio
async def test_verify_api_key_invalid_token(auth_module, monkeypatch):
    monkeypatch.setenv("WORKER_API_KEY", "secret-key")
    request = MagicMock()
    # Correct scheme, wrong token
    request.headers.get.return_value = "Bearer wrong-key"
    request.client = "test-client"

    with pytest.raises(MockHTTPError) as excinfo:
        await auth_module.verify_api_key(request)

    assert excinfo.value.status_code == 401
    assert excinfo.value.detail == "Invalid API key"

@pytest.mark.asyncio
async def test_verify_api_key_success(auth_module, monkeypatch):
    monkeypatch.setenv("WORKER_API_KEY", "secret-key")
    request = MagicMock()
    # Correct scheme and token
    request.headers.get.return_value = "Bearer secret-key"

    # Should not raise exception
    result = await auth_module.verify_api_key(request)
    assert result is None
