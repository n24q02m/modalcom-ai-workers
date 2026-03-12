"""Tests for RerankerServer FastAPI routes."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from ai_workers.workers.reranker import MODEL_CONFIGS, RerankerServer


@pytest.fixture()
def server():
    return RerankerServer()


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


def test_rerank_requires_auth(server):
    app = server.serve()
    tc = TestClient(app)
    resp = tc.post(
        "/v1/rerank",
        json={"model": "qwen3-reranker-0.6b", "query": "q", "documents": ["d1"]},
    )
    assert resp.status_code == 401


def test_rerank_unknown_model(server):
    with patch.dict(os.environ, {"API_KEY": "k"}):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/rerank",
            json={"model": "bad-model", "query": "q", "documents": ["d1"]},
            headers={"Authorization": "Bearer k"},
        )
    assert resp.status_code == 400
    assert "Unknown model" in resp.json()["error"]


# ---------------------------------------------------------------------------
# /v1/rerank — valid requests
# ---------------------------------------------------------------------------


def test_rerank_single_document(server):
    server._score_pairs = MagicMock(return_value=[0.9])

    with patch.dict(os.environ, {"API_KEY": "k"}):
        import ai_workers.common.auth as auth_mod
        auth_mod._valid_keys = None
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/rerank",
            json={
                "model": "qwen3-reranker-0.6b",
                "query": "What is Python?",
                "documents": ["Python is a language."],
                "return_documents": True,
            },
            headers={"Authorization": "Bearer k"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "qwen3-reranker-0.6b"
    assert len(data["results"]) == 1
    assert data["results"][0]["relevance_score"] == pytest.approx(0.9)
    assert data["results"][0]["document"]["text"] == "Python is a language."


def test_rerank_multiple_documents_sorted(server):
    scores = [0.3, 0.9, 0.5]
    server._score_pairs = MagicMock(return_value=scores)

    with patch.dict(os.environ, {"API_KEY": "k"}):
        import ai_workers.common.auth as auth_mod
        auth_mod._valid_keys = None
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/rerank",
            json={
                "model": "qwen3-reranker-0.6b",
                "query": "query",
                "documents": ["doc0", "doc1", "doc2"],
            },
            headers={"Authorization": "Bearer k"},
        )

    results = resp.json()["results"]
    # Should be sorted descending by score
    relevance_scores = [r["relevance_score"] for r in results]
    assert relevance_scores == sorted(relevance_scores, reverse=True)


def test_rerank_top_n(server):
    server._score_pairs = MagicMock(return_value=[0.3, 0.9, 0.5])

    with patch.dict(os.environ, {"API_KEY": "k"}):
        import ai_workers.common.auth as auth_mod
        auth_mod._valid_keys = None
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/rerank",
            json={
                "model": "qwen3-reranker-0.6b",
                "query": "query",
                "documents": ["d0", "d1", "d2"],
                "top_n": 2,
            },
            headers={"Authorization": "Bearer k"},
        )

    assert len(resp.json()["results"]) == 2


def test_rerank_heavy_model(server):
    server._score_pairs = MagicMock(return_value=[0.7])

    with patch.dict(os.environ, {"API_KEY": "k"}):
        import ai_workers.common.auth as auth_mod
        auth_mod._valid_keys = None
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/rerank",
            json={
                "model": "qwen3-reranker-8b",
                "query": "q",
                "documents": ["d"],
            },
            headers={"Authorization": "Bearer k"},
        )

    assert resp.status_code == 200
    assert resp.json()["model"] == "qwen3-reranker-8b"
