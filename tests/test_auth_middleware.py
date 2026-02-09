import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_workers.common.auth import auth_middleware


@pytest.fixture
def client():
    app = FastAPI()
    app.middleware("http")(auth_middleware)

    @app.get("/")
    def root():
        return {"message": "root"}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/protected")
    def protected():
        return {"message": "protected"}

    return TestClient(app, raise_server_exceptions=False)


def test_health_check_no_auth(client):
    """Health check should not require authentication."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_no_auth(client):
    """Root path should not require authentication."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "root"}


def test_protected_no_auth_dev_mode(client, monkeypatch):
    """In dev mode (no WORKER_API_KEY), protected routes should be accessible."""
    monkeypatch.delenv("WORKER_API_KEY", raising=False)
    response = client.get("/protected")
    assert response.status_code == 200
    assert response.json() == {"message": "protected"}


def test_protected_missing_auth(client, monkeypatch):
    """If WORKER_API_KEY is set, missing auth header should return 401."""
    monkeypatch.setenv("WORKER_API_KEY", "secret-key")
    response = client.get("/protected")
    assert response.status_code == 401
    assert response.json()["detail"] == "Missing Bearer token"


def test_protected_invalid_auth(client, monkeypatch):
    """If WORKER_API_KEY is set, invalid auth header should return 401."""
    monkeypatch.setenv("WORKER_API_KEY", "secret-key")
    response = client.get("/protected", headers={"Authorization": "Bearer wrong-key"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API key"


def test_protected_valid_auth(client, monkeypatch):
    """If WORKER_API_KEY is set, valid auth header should return 200."""
    monkeypatch.setenv("WORKER_API_KEY", "secret-key")
    response = client.get("/protected", headers={"Authorization": "Bearer secret-key"})
    assert response.status_code == 200
    assert response.json() == {"message": "protected"}


def test_malformed_auth_header(client, monkeypatch):
    """Malformed auth header (not Bearer) should return 401."""
    monkeypatch.setenv("WORKER_API_KEY", "secret-key")
    response = client.get("/protected", headers={"Authorization": "Basic user:pass"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Missing Bearer token"
