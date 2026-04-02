"""Tests for VLRerankerServer FastAPI routes."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from ai_workers.workers.vl_reranker import MODEL_CONFIGS, VLRerankerServer


@pytest.fixture()
def server():
    s = VLRerankerServer()
    s.models = {}
    s.processors = {}
    return s


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


def test_rerank_requires_auth(server):
    app = server.serve()
    tc = TestClient(app)
    resp = tc.post(
        "/v1/rerank",
        json={"model": "qwen3-vl-reranker-8b", "query": "q", "documents": ["d"]},
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Unknown model
# ---------------------------------------------------------------------------


def test_rerank_unknown_model(server):
    with patch.dict(os.environ, {"API_KEY": "k"}):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/rerank",
            json={"model": "bad-model", "query": "q", "documents": ["d"]},
            headers={"Authorization": "Bearer k"},
        )
    assert resp.status_code == 400
    assert "Unknown model" in resp.json()["error"]


# ---------------------------------------------------------------------------
# /v1/rerank — text-only docs (list[str])
# ---------------------------------------------------------------------------


def test_rerank_text_only_docs(server):
    server._score_pair = MagicMock(return_value=0.8)
    server._load_image = MagicMock(return_value="mock_img")

    with patch.dict(os.environ, {"API_KEY": "k"}):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/rerank",
            json={
                "model": "qwen3-vl-reranker-8b",
                "query": "What is AI?",
                "documents": ["AI is artificial intelligence.", "ML is machine learning."],
            },
            headers={"Authorization": "Bearer k"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "qwen3-vl-reranker-8b"
    assert len(data["results"]) == 2
    # Verify _score_pair called with text-only (no image URLs)
    for call_args in server._score_pair.call_args_list:
        assert call_args.kwargs.get("query_image") is None
        assert call_args.kwargs.get("document_image") is None


# ---------------------------------------------------------------------------
# /v1/rerank — multimodal docs (list[VLRerankDocument])
# ---------------------------------------------------------------------------


def test_rerank_multimodal_docs(server):
    server._score_pair = MagicMock(side_effect=[0.9, 0.4])
    server._load_image = MagicMock(return_value="mock_img")

    with patch.dict(os.environ, {"API_KEY": "k"}):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/rerank",
            json={
                "model": "qwen3-vl-reranker-8b",
                "query": "describe",
                "documents": [
                    {"text": "doc with image", "image_url": "http://example.com/img.jpg"},
                    {"text": "doc without image"},
                ],
            },
            headers={"Authorization": "Bearer k"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 2
    # First call should have document_image_url set
    first_call = server._score_pair.call_args_list[0]
    assert first_call.kwargs.get("document_image") == "mock_img"
    # Second call should have no document_image_url
    second_call = server._score_pair.call_args_list[1]
    assert second_call.kwargs.get("document_image") is None


# ---------------------------------------------------------------------------
# /v1/rerank — sorted descending
# ---------------------------------------------------------------------------


def test_rerank_sorted_descending(server):
    server._score_pair = MagicMock(side_effect=[0.3, 0.9, 0.5])
    server._load_image = MagicMock(return_value="mock_img")

    with patch.dict(os.environ, {"API_KEY": "k"}):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/rerank",
            json={
                "model": "qwen3-vl-reranker-8b",
                "query": "query",
                "documents": ["d0", "d1", "d2"],
            },
            headers={"Authorization": "Bearer k"},
        )

    results = resp.json()["results"]
    scores = [r["relevance_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# /v1/rerank — top_n
# ---------------------------------------------------------------------------


def test_rerank_top_n(server):
    server._score_pair = MagicMock(side_effect=[0.3, 0.9, 0.5])
    server._load_image = MagicMock(return_value="mock_img")

    with patch.dict(os.environ, {"API_KEY": "k"}):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/rerank",
            json={
                "model": "qwen3-vl-reranker-8b",
                "query": "query",
                "documents": ["d0", "d1", "d2"],
                "top_n": 2,
            },
            headers={"Authorization": "Bearer k"},
        )

    assert len(resp.json()["results"]) == 2


# ---------------------------------------------------------------------------
# /v1/rerank — with query_image_url
# ---------------------------------------------------------------------------


def test_rerank_with_query_image_url(server):
    server._score_pair = MagicMock(return_value=0.7)
    server._load_image = MagicMock(return_value="mock_img")

    with patch.dict(os.environ, {"API_KEY": "k"}):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/rerank",
            json={
                "model": "qwen3-vl-reranker-8b",
                "query": "q",
                "query_image_url": "http://example.com/query.jpg",
                "documents": ["d"],
            },
            headers={"Authorization": "Bearer k"},
        )

    assert resp.status_code == 200
    call_args = server._score_pair.call_args
    assert call_args.kwargs.get("query_image") == "mock_img"


# ---------------------------------------------------------------------------
# /v1/rerank — heavy model (8b)
# ---------------------------------------------------------------------------


def test_rerank_heavy_model(server):
    server._score_pair = MagicMock(return_value=0.6)
    server._load_image = MagicMock(return_value="mock_img")

    with patch.dict(os.environ, {"API_KEY": "k"}):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/rerank",
            json={
                "model": "qwen3-vl-reranker-8b",
                "query": "q",
                "documents": ["d"],
            },
            headers={"Authorization": "Bearer k"},
        )

    assert resp.status_code == 200
    assert resp.json()["model"] == "qwen3-vl-reranker-8b"


# ---------------------------------------------------------------------------
# _load_image error handling
# ---------------------------------------------------------------------------


def test_load_image_error(server):
    with (
        patch(
            "ai_workers.common.utils.is_safe_url",
            return_value=True,
        ),
        patch(
            "requests.get",
            side_effect=Exception("Connection error"),
        ),
        pytest.raises(ValueError, match=r"URL blocked by SSRF protection: http://bad.url"),
    ):
        server._load_image("http://bad.url")
