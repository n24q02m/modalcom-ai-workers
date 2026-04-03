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
    server._embed_multimodal = MagicMock(return_value=[0.5, 0.6, 0.7])

    with patch.dict(os.environ, {"API_KEY": "k"}):
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
    server._embed_multimodal.assert_called_once_with(
        "qwen3-vl-embedding-2b", "describe this image", "http://example.com/img.jpg"
    )


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
    server._embed_multimodal = MagicMock(return_value=[0.9, 0.8])
    server._embed_text = MagicMock(return_value=[[0.1, 0.2]])

    with patch.dict(os.environ, {"API_KEY": "k"}):
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
    mock_qwen_vl_utils = MagicMock()
    mock_qwen_vl_utils.process_vision_info.side_effect = ValueError("Failed to load image")

    with (
        patch.dict(os.environ, {"API_KEY": "k"}),
        patch.dict("sys.modules", {"qwen_vl_utils": mock_qwen_vl_utils}),
    ):
        app = server.serve()
        # Ensure raise_server_exceptions is False so it returns a 500 status code
        tc = TestClient(app, raise_server_exceptions=False)

        # _embed_multimodal needs actual server structure for this test to reach process_vision_info
        server.models = {"qwen3-vl-embedding-2b": MagicMock()}

        mock_processor = MagicMock()
        mock_processor.apply_chat_template.return_value = "chat_text"
        server.processors = {"qwen3-vl-embedding-2b": mock_processor}

        resp = tc.post(
            "/v1/embeddings",
            json={
                "model": "qwen3-vl-embedding-2b",
                "input": {"text": "describe this image", "image_url": "http://example.com/bad.jpg"},
            },
            headers={"Authorization": "Bearer k"},
        )

    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# Batched inference verification
# ---------------------------------------------------------------------------


def test_embed_text_is_batched(server):
    mock_model = MagicMock()
    mock_processor = MagicMock()

    server.models = {"qwen3-vl-embedding-2b": mock_model}
    server.processors = {"qwen3-vl-embedding-2b": mock_processor}

    mock_model.device = "cpu"
    # Mock apply_chat_template to return a list of formatted strings
    mock_processor.apply_chat_template.return_value = ["formatted1", "formatted2"]

    # Mock processor return value to support .to(device) and behaving like a dict
    mock_inputs = {"attention_mask": MagicMock()}
    mock_processor.return_value.to.return_value = mock_inputs

    # Mock model output
    mock_outputs = MagicMock()
    mock_model.return_value = mock_outputs

    # Mock _last_token_pool and torch.nn.functional.normalize
    # Use patch context managers since these are called inside _embed_text
    with (
        patch.object(server, "_last_token_pool") as mock_pool,
        patch("torch.nn.functional.normalize") as mock_norm,
    ):
        mock_batched_embeddings = MagicMock()
        mock_pool.return_value = mock_batched_embeddings
        mock_norm.return_value = mock_batched_embeddings

        # Mock the slicing and conversion to list
        # embeddings[:, :EMBEDDING_DIM].cpu().tolist()
        mock_batched_embeddings.__getitem__.return_value = mock_batched_embeddings
        mock_batched_embeddings.cpu.return_value.tolist.return_value = [
            [0.1] * 1024,
            [0.1] * 1024,
        ]

        embeddings = server._embed_text("qwen3-vl-embedding-2b", ["text1", "text2"])

        # Assertions
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 1024

        # Verify batched calls
        mock_model.assert_called_once()
        mock_processor.apply_chat_template.assert_called_once()
        # Verify it was called with the whole list
        called_args = mock_processor.apply_chat_template.call_args[0][0]
        assert len(called_args) == 2

        # Verify processor was called with the strings from apply_chat_template
        mock_processor.assert_called_once()
        assert mock_processor.call_args[1]["text"] == ["formatted1", "formatted2"]
