import os
import sys
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Ensure ai_workers is in python path
sys.path.insert(0, "src")

from ai_workers.common.auth import auth_middleware


@pytest.fixture
def app():
    app = FastAPI()
    app.middleware("http")(auth_middleware)

    @app.get("/")
    def root():
        return {"message": "root"}

    @app.get("/health")
    def health():
        return {"message": "health"}

    @app.get("/protected")
    def protected():
        return {"message": "protected"}

    return app


def test_health_no_auth(app):
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200


def test_root_no_auth(app):
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200


def test_protected_no_auth_header(app):
    with patch.dict(os.environ, {"WORKER_API_KEY": "secret"}):
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/protected")
        assert response.status_code == 401
        assert response.json()["detail"] == "Missing Bearer token"


def test_protected_invalid_auth_header(app):
    with patch.dict(os.environ, {"WORKER_API_KEY": "secret"}):
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/protected", headers={"Authorization": "Bearer wrong"})
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid API key"


def test_protected_valid_auth_header(app):
    with patch.dict(os.environ, {"WORKER_API_KEY": "secret"}):
        client = TestClient(app)
        response = client.get("/protected", headers={"Authorization": "Bearer secret"})
        assert response.status_code == 200


def test_protected_no_env_var(app):
    with patch.dict(os.environ, {"WORKER_API_KEY": ""}):
        client = TestClient(app)
        response = client.get("/protected")
        assert response.status_code == 200
