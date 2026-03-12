"""Tests for EmbeddingServer FastAPI routes."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from ai_workers.workers.embedding import MODEL_CONFIGS, EmbeddingServer


@pytest.fixture()
def server():
    return EmbeddingServer()


@pytest.fixture()
def client(server):
    app = server.serve()
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture()
def authed_client(server):
    """Client with API key set in environment."""
    with patch.dict(os.environ, {"API_KEY": "test-secret-key"}):
        app = server.serve()
        return TestClient(app, raise_server_exceptions=True), "test-secret-key"


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


def test_health_no_auth(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert set(data["models"]) == set(MODEL_CONFIGS.keys())


def test_health_returns_model_names(client):
    resp = client.get("/health")
    assert "qwen3-embedding-0.6b" in resp.json()["models"]
    assert "qwen3-embedding-8b" in resp.json()["models"]


# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------


def test_embeddings_requires_auth(client):
    resp = client.post(
        "/v1/embeddings",
        json={"model": "qwen3-embedding-0.6b", "input": "hello"},
    )
    assert resp.status_code == 401


def test_embeddings_with_valid_key(server):
    with patch.dict(os.environ, {"API_KEY": "my-key"}):
        # Reset the cached keys in auth module so it reads the new patch
        import ai_workers.common.auth as auth_mod

        auth_mod._valid_keys = None

        app = server.serve()
        # Mock _embed so we don't need real torch
        fake_emb = [[0.1] * 10]
        server._embed = MagicMock(return_value=fake_emb)
        # Mock tokenizer encode
        mock_tok = MagicMock()
        mock_tok.encode.return_value = [1, 2, 3]
        server.tokenizers = {"qwen3-embedding-0.6b": mock_tok}

        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/embeddings",
            json={"model": "qwen3-embedding-0.6b", "input": "hello"},
            headers={"Authorization": "Bearer my-key"},
        )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /v1/embeddings — invalid model
# ---------------------------------------------------------------------------


def test_embeddings_unknown_model(server):
    with patch.dict(os.environ, {"API_KEY": "k"}):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/embeddings",
            json={"model": "nonexistent-model", "input": "hello"},
            headers={"Authorization": "Bearer k"},
        )
    assert resp.status_code == 400
    assert "Unknown model" in resp.json()["error"]


# ---------------------------------------------------------------------------
# /v1/embeddings — valid requests
# ---------------------------------------------------------------------------


def test_embeddings_string_input(server):
    fake_emb = [[0.1, 0.2, 0.3]]
    server._embed = MagicMock(return_value=fake_emb)
    mock_tok = MagicMock()
    mock_tok.encode.return_value = [1, 2]
    server.tokenizers = {"qwen3-embedding-0.6b": mock_tok}

    with patch.dict(os.environ, {"API_KEY": "k"}):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/embeddings",
            json={"model": "qwen3-embedding-0.6b", "input": "hello world"},
            headers={"Authorization": "Bearer k"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["embedding"] == [0.1, 0.2, 0.3]
    assert data["data"][0]["index"] == 0
    assert data["model"] == "qwen3-embedding-0.6b"
    assert "usage" in data


def test_embeddings_list_input(server):
    fake_emb = [[0.1, 0.2], [0.3, 0.4]]
    server._embed = MagicMock(return_value=fake_emb)
    mock_tok = MagicMock()
    mock_tok.encode.return_value = [1]
    server.tokenizers = {"qwen3-embedding-0.6b": mock_tok}

    with patch.dict(os.environ, {"API_KEY": "k"}):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/embeddings",
            json={"model": "qwen3-embedding-0.6b", "input": ["text1", "text2"]},
            headers={"Authorization": "Bearer k"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["data"]) == 2
    assert data["data"][1]["index"] == 1


def test_embeddings_usage_tokens(server):
    """Usage field should reflect sum of tokenizer encode lengths."""
    server._embed = MagicMock(return_value=[[0.0, 0.0]])
    mock_tok = MagicMock()
    mock_tok.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
    server.tokenizers = {"qwen3-embedding-0.6b": mock_tok}

    with patch.dict(os.environ, {"API_KEY": "k"}):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/embeddings",
            json={"model": "qwen3-embedding-0.6b", "input": "hello"},
            headers={"Authorization": "Bearer k"},
        )

    usage = resp.json()["usage"]
    assert usage["prompt_tokens"] == 5
    assert usage["total_tokens"] == 5


def test_embeddings_heavy_model(server):
    fake_emb = [[0.5] * 5]
    server._embed = MagicMock(return_value=fake_emb)
    mock_tok = MagicMock()
    mock_tok.encode.return_value = [1]
    server.tokenizers = {"qwen3-embedding-8b": mock_tok}

    with patch.dict(os.environ, {"API_KEY": "k"}):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/embeddings",
            json={"model": "qwen3-embedding-8b", "input": "test"},
            headers={"Authorization": "Bearer k"},
        )

    assert resp.status_code == 200
    assert resp.json()["model"] == "qwen3-embedding-8b"
