"""Tests for VLEmbeddingServer FastAPI routes."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from ai_workers.workers.vl_embedding import MODEL_CONFIGS, VLEmbeddingServer


@pytest.fixture()
def server():
    s = VLEmbeddingServer()
    s.models = {}
    s.processors = {}
    return s


def _make_client(server, api_key="k"):
    with patch.dict(os.environ, {"API_KEY": api_key}):
        app = server.serve()
    return TestClient(app, raise_server_exceptions=True), api_key


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health(server):
    app = server.serve()
    tc = TestClient(app)
    resp = tc.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert set(resp.json()["models"]) == set(MODEL_CONFIGS.keys())


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_embeddings_requires_auth(server):
    app = server.serve()
    tc = TestClient(app)
    resp = tc.post(
        "/v1/embeddings",
        json={"model": "qwen3-vl-embedding-2b", "input": "hello"},
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Unknown model
# ---------------------------------------------------------------------------


def test_embeddings_unknown_model(server):
    with patch.dict(os.environ, {"API_KEY": "k"}):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/embeddings",
            json={"model": "bad-model", "input": "hello"},
            headers={"Authorization": "Bearer k"},
        )
    assert resp.status_code == 400
    assert "Unknown model" in resp.json()["error"]


# ---------------------------------------------------------------------------
# /v1/embeddings — string input
# ---------------------------------------------------------------------------


def test_embeddings_string_input(server):
    server._embed_text = MagicMock(return_value=[[0.1, 0.2, 0.3]])

    with patch.dict(os.environ, {"API_KEY": "k"}):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/embeddings",
            json={"model": "qwen3-vl-embedding-2b", "input": "hello world"},
            headers={"Authorization": "Bearer k"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "qwen3-vl-embedding-2b"
    assert len(data["data"]) == 1
    assert data["data"][0]["embedding"] == [0.1, 0.2, 0.3]
    assert data["data"][0]["index"] == 0
    server._embed_text.assert_called_once_with("qwen3-vl-embedding-2b", ["hello world"])


# ---------------------------------------------------------------------------
# /v1/embeddings — list[str] input
# ---------------------------------------------------------------------------


def test_embeddings_list_of_strings(server):
    server._embed_text = MagicMock(return_value=[[0.1, 0.2], [0.3, 0.4]])

    with patch.dict(os.environ, {"API_KEY": "k"}):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/embeddings",
            json={"model": "qwen3-vl-embedding-2b", "input": ["text1", "text2"]},
            headers={"Authorization": "Bearer k"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["data"]) == 2
    assert data["data"][0]["index"] == 0
    assert data["data"][1]["index"] == 1
    server._embed_text.assert_called_once_with("qwen3-vl-embedding-2b", ["text1", "text2"])


# ---------------------------------------------------------------------------
# /v1/embeddings — VLEmbeddingInput with image_url
# ---------------------------------------------------------------------------


def test_embeddings_vlinput_with_image_url(server):
    server._embed_multimodal = MagicMock(return_value=[[0.5, 0.6, 0.7]])

    async def mock_load_image(url):
        return MagicMock()

    with (
        patch.dict(os.environ, {"API_KEY": "k"}),
        patch.object(server, '_load_image_from_url', return_value=MagicMock())
    ):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/embeddings",
            json={
                "model": "qwen3-vl-embedding-2b",
                "input": {"text": "describe this image", "image_url": "http://example.com/img.jpg"},
            },
            headers={"Authorization": "Bearer k"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["data"]) == 1
    assert data["data"][0]["embedding"] == [0.5, 0.6, 0.7]

    server._embed_multimodal.assert_called_once()
    args = server._embed_multimodal.call_args[0]
    assert args[0] == "qwen3-vl-embedding-2b"
    assert args[1] == ["describe this image"]
    # It passes the MagicMock image object now, not the URL string
    assert len(args[2]) == 1


# ---------------------------------------------------------------------------
# /v1/embeddings — VLEmbeddingInput without image_url (text-only)
# ---------------------------------------------------------------------------


def test_embeddings_vlinput_without_image_url(server):
    server._embed_text = MagicMock(return_value=[[0.1, 0.2]])

    with patch.dict(os.environ, {"API_KEY": "k"}):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/embeddings",
            json={
                "model": "qwen3-vl-embedding-2b",
                "input": {"text": "text only"},
            },
            headers={"Authorization": "Bearer k"},
        )

    assert resp.status_code == 200
    server._embed_text.assert_called_once_with("qwen3-vl-embedding-2b", ["text only"])


# ---------------------------------------------------------------------------
# /v1/embeddings — list[VLEmbeddingInput] mixed
# ---------------------------------------------------------------------------


def test_embeddings_list_of_vlinputs(server):
    server._embed_multimodal = MagicMock(return_value=[[0.9, 0.8]])
    server._embed_text = MagicMock(return_value=[[0.1, 0.2]])

    with (
        patch.dict(os.environ, {"API_KEY": "k"}),
        patch.object(server, '_load_image_from_url', return_value=MagicMock())
    ):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/embeddings",
            json={
                "model": "qwen3-vl-embedding-2b",
                "input": [
                    {"text": "img text", "image_url": "http://example.com/img.jpg"},
                    {"text": "no image"},
                ],
            },
            headers={"Authorization": "Bearer k"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["data"]) == 2
    assert data["data"][0]["embedding"] == [0.9, 0.8]
    assert data["data"][1]["embedding"] == [0.1, 0.2]

    server._embed_multimodal.assert_called_once()
    args = server._embed_multimodal.call_args[0]
    assert args[0] == "qwen3-vl-embedding-2b"
    assert args[1] == ["img text"]
    assert len(args[2]) == 1

    server._embed_text.assert_called_once_with("qwen3-vl-embedding-2b", ["no image"])


# ---------------------------------------------------------------------------
# /v1/embeddings — multiple multimodal inputs (batching test)
# ---------------------------------------------------------------------------


def test_embeddings_multiple_multimodal_batching(server):
    server._embed_multimodal = MagicMock(return_value=[[0.1, 0.1], [0.2, 0.2]])

    with (
        patch.dict(os.environ, {"API_KEY": "k"}),
        patch.object(server, '_load_image_from_url', return_value=MagicMock())
    ):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/embeddings",
            json={
                "model": "qwen3-vl-embedding-2b",
                "input": [
                    {"text": "text1", "image_url": "http://example.com/img1.jpg"},
                    {"text": "text2", "image_url": "http://example.com/img2.jpg"},
                ],
            },
            headers={"Authorization": "Bearer k"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["data"]) == 2
    assert data["data"][0]["embedding"] == [0.1, 0.1]
    assert data["data"][1]["embedding"] == [0.2, 0.2]

    server._embed_multimodal.assert_called_once()
    args = server._embed_multimodal.call_args[0]
    assert args[0] == "qwen3-vl-embedding-2b"
    assert args[1] == ["text1", "text2"]
    assert len(args[2]) == 2


# ---------------------------------------------------------------------------
# /v1/embeddings — heavy model (8b)
# ---------------------------------------------------------------------------


def test_embeddings_heavy_model(server):
    server._embed_text = MagicMock(return_value=[[0.1] * 10])

    with patch.dict(os.environ, {"API_KEY": "k"}):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/embeddings",
            json={"model": "qwen3-vl-embedding-8b", "input": "test"},
            headers={"Authorization": "Bearer k"},
        )

    assert resp.status_code == 200
    assert resp.json()["model"] == "qwen3-vl-embedding-8b"


# ---------------------------------------------------------------------------
# /v1/embeddings — VLEmbeddingInput with image fetch failure
# ---------------------------------------------------------------------------


def test_embeddings_vlinput_image_fetch_failure(server):
    with (
        patch.dict(os.environ, {"API_KEY": "k"}),
        patch.object(server, '_load_image_from_url', side_effect=ValueError("Failed to load image"))
    ):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=False)

        resp = tc.post(
            "/v1/embeddings",
            json={
                "model": "qwen3-vl-embedding-2b",
                "input": {"text": "describe this image", "image_url": "http://example.com/bad.jpg"},
            },
            headers={"Authorization": "Bearer k"},
        )

    assert resp.status_code == 400
