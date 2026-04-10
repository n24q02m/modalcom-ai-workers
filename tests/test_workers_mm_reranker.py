"""Tests for MmRerankerServer FastAPI routes.

Covers text-only, image, audio, video, mixed modality inputs,
API versioning (/v1, /v2), auth, validation, sorting, top_n,
return_documents, and error handling.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from ai_workers.workers.mm_reranker import MODEL_CONFIGS, MmRerankerServer


@pytest.fixture()
def server():
    s = MmRerankerServer()
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


def test_health_model_list(server):
    app = server.serve()
    tc = TestClient(app)
    resp = tc.get("/health")
    assert set(resp.json()["models"]) == set(MODEL_CONFIGS.keys())


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_rerank_requires_auth(server):
    app = server.serve()
    tc = TestClient(app)
    resp = tc.post(
        "/v1/rerank",
        json={"model": "gemma4-reranker-v1", "query": "q", "documents": ["d"]},
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
# Text-only
# ---------------------------------------------------------------------------


def test_rerank_text_only(server):
    server._score_pair = MagicMock(return_value=0.8)

    with patch.dict(os.environ, {"API_KEY": "k"}):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/rerank",
            json={
                "model": "gemma4-reranker-v1",
                "query": "What is AI?",
                "documents": ["AI is artificial intelligence.", "ML is machine learning."],
            },
            headers={"Authorization": "Bearer k"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "gemma4-reranker-v1"
    assert len(data["results"]) == 2
    # All media params should be None for text-only
    for call_args in server._score_pair.call_args_list:
        assert call_args.kwargs.get("query_image_url") is None
        assert call_args.kwargs.get("query_audio_url") is None
        assert call_args.kwargs.get("query_video_url") is None
        assert call_args.kwargs.get("doc_image_url") is None
        assert call_args.kwargs.get("doc_audio_url") is None
        assert call_args.kwargs.get("doc_video_url") is None


def test_rerank_sorted_descending(server):
    server._score_pair = MagicMock(side_effect=[0.3, 0.9, 0.5])

    with patch.dict(os.environ, {"API_KEY": "k"}):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/rerank",
            json={
                "model": "gemma4-reranker-v1",
                "query": "query",
                "documents": ["d0", "d1", "d2"],
            },
            headers={"Authorization": "Bearer k"},
        )

    results = resp.json()["results"]
    scores = [r["relevance_score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    # Verify original indices preserved
    assert results[0]["index"] == 1  # score 0.9
    assert results[1]["index"] == 2  # score 0.5
    assert results[2]["index"] == 0  # score 0.3


def test_rerank_top_n(server):
    server._score_pair = MagicMock(side_effect=[0.3, 0.9, 0.5])

    with patch.dict(os.environ, {"API_KEY": "k"}):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/rerank",
            json={
                "model": "gemma4-reranker-v1",
                "query": "query",
                "documents": ["d0", "d1", "d2"],
                "top_n": 2,
            },
            headers={"Authorization": "Bearer k"},
        )

    assert len(resp.json()["results"]) == 2


def test_rerank_return_documents(server):
    server._score_pair = MagicMock(return_value=0.7)

    with patch.dict(os.environ, {"API_KEY": "k"}):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/rerank",
            json={
                "model": "gemma4-reranker-v1",
                "query": "q",
                "documents": ["doc text here"],
                "return_documents": True,
            },
            headers={"Authorization": "Bearer k"},
        )

    result = resp.json()["results"][0]
    assert result["document"] is not None
    assert result["document"]["text"] == "doc text here"


def test_rerank_no_return_documents_by_default(server):
    server._score_pair = MagicMock(return_value=0.7)

    with patch.dict(os.environ, {"API_KEY": "k"}):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/rerank",
            json={
                "model": "gemma4-reranker-v1",
                "query": "q",
                "documents": ["doc text"],
            },
            headers={"Authorization": "Bearer k"},
        )

    result = resp.json()["results"][0]
    assert result["document"] is None


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------


def test_rerank_with_query_image(server):
    server._score_pair = MagicMock(return_value=0.7)

    with patch.dict(os.environ, {"API_KEY": "k"}):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/rerank",
            json={
                "model": "gemma4-reranker-v1",
                "query": "q",
                "query_image": "http://example.com/query.jpg",
                "documents": ["d"],
            },
            headers={"Authorization": "Bearer k"},
        )

    assert resp.status_code == 200
    call_args = server._score_pair.call_args
    assert call_args.args[1].image_url == "http://example.com/query.jpg"


def test_rerank_with_doc_images(server):
    server._score_pair = MagicMock(side_effect=[0.9, 0.4])

    with patch.dict(os.environ, {"API_KEY": "k"}):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/rerank",
            json={
                "model": "gemma4-reranker-v1",
                "query": "describe",
                "documents": ["doc1", "doc2"],
                "doc_images": ["http://example.com/img1.jpg", None],
            },
            headers={"Authorization": "Bearer k"},
        )

    assert resp.status_code == 200
    calls = server._score_pair.call_args_list
    assert calls[0].args[2].image_url == "http://example.com/img1.jpg"
    assert calls[1].args[2].image_url is None


def test_rerank_doc_images_length_mismatch(server):
    with patch.dict(os.environ, {"API_KEY": "k"}):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/rerank",
            json={
                "model": "gemma4-reranker-v1",
                "query": "q",
                "documents": ["d1", "d2"],
                "doc_images": ["http://example.com/img.jpg"],
            },
            headers={"Authorization": "Bearer k"},
        )

    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------


def test_rerank_with_query_audio(server):
    server._score_pair = MagicMock(return_value=0.6)

    with patch.dict(os.environ, {"API_KEY": "k"}):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/rerank",
            json={
                "model": "gemma4-reranker-v1",
                "query": "speech about AI",
                "query_audio": "http://example.com/query.wav",
                "documents": ["AI document"],
            },
            headers={"Authorization": "Bearer k"},
        )

    assert resp.status_code == 200
    call_args = server._score_pair.call_args
    assert call_args.args[1].audio_url == "http://example.com/query.wav"


def test_rerank_with_doc_audios(server):
    server._score_pair = MagicMock(side_effect=[0.8, 0.3])

    with patch.dict(os.environ, {"API_KEY": "k"}):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/rerank",
            json={
                "model": "gemma4-reranker-v1",
                "query": "q",
                "documents": ["d1", "d2"],
                "doc_audios": ["http://example.com/a1.wav", None],
            },
            headers={"Authorization": "Bearer k"},
        )

    assert resp.status_code == 200
    calls = server._score_pair.call_args_list
    assert calls[0].args[2].audio_url == "http://example.com/a1.wav"
    assert calls[1].args[2].audio_url is None


def test_rerank_doc_audios_length_mismatch(server):
    with patch.dict(os.environ, {"API_KEY": "k"}):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/rerank",
            json={
                "model": "gemma4-reranker-v1",
                "query": "q",
                "documents": ["d1", "d2"],
                "doc_audios": ["http://example.com/a.wav"],
            },
            headers={"Authorization": "Bearer k"},
        )

    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------


def test_rerank_with_query_video(server):
    server._score_pair = MagicMock(return_value=0.5)

    with patch.dict(os.environ, {"API_KEY": "k"}):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/rerank",
            json={
                "model": "gemma4-reranker-v1",
                "query": "activity recognition",
                "query_video": "http://example.com/video.mp4",
                "documents": ["running activity"],
            },
            headers={"Authorization": "Bearer k"},
        )

    assert resp.status_code == 200
    call_args = server._score_pair.call_args
    assert call_args.args[1].video_url == "http://example.com/video.mp4"


def test_rerank_with_doc_videos(server):
    server._score_pair = MagicMock(side_effect=[0.7, 0.2])

    with patch.dict(os.environ, {"API_KEY": "k"}):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/rerank",
            json={
                "model": "gemma4-reranker-v1",
                "query": "q",
                "documents": ["d1", "d2"],
                "doc_videos": ["http://example.com/v1.mp4", None],
            },
            headers={"Authorization": "Bearer k"},
        )

    assert resp.status_code == 200
    calls = server._score_pair.call_args_list
    assert calls[0].args[2].video_url == "http://example.com/v1.mp4"
    assert calls[1].args[2].video_url is None


def test_rerank_doc_videos_length_mismatch(server):
    with patch.dict(os.environ, {"API_KEY": "k"}):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/rerank",
            json={
                "model": "gemma4-reranker-v1",
                "query": "q",
                "documents": ["d1"],
                "doc_videos": ["http://a.mp4", "http://b.mp4"],
            },
            headers={"Authorization": "Bearer k"},
        )

    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Mixed modalities
# ---------------------------------------------------------------------------


def test_rerank_mixed_modalities(server):
    """Test request with image, audio, and video together."""
    server._score_pair = MagicMock(return_value=0.85)

    with patch.dict(os.environ, {"API_KEY": "k"}):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/rerank",
            json={
                "model": "gemma4-reranker-v1",
                "query": "multimodal query",
                "query_image": "http://example.com/qi.jpg",
                "query_audio": "http://example.com/qa.wav",
                "query_video": "http://example.com/qv.mp4",
                "documents": ["multimodal doc"],
                "doc_images": ["http://example.com/di.jpg"],
                "doc_audios": ["http://example.com/da.wav"],
                "doc_videos": ["http://example.com/dv.mp4"],
            },
            headers={"Authorization": "Bearer k"},
        )

    assert resp.status_code == 200
    call_args = server._score_pair.call_args
    assert call_args.args[1].image_url == "http://example.com/qi.jpg"
    assert call_args.args[1].audio_url == "http://example.com/qa.wav"
    assert call_args.args[1].video_url == "http://example.com/qv.mp4"
    assert call_args.args[2].image_url == "http://example.com/di.jpg"
    assert call_args.args[2].audio_url == "http://example.com/da.wav"
    assert call_args.args[2].video_url == "http://example.com/dv.mp4"


# ---------------------------------------------------------------------------
# Scoring error handling
# ---------------------------------------------------------------------------


def test_rerank_score_pair_value_error(server):
    """ValueError from _score_pair (e.g. audio > 30s) should return 400."""
    server._score_pair = MagicMock(
        side_effect=ValueError("Audio duration 45.0s exceeds maximum 30.0s")
    )

    with patch.dict(os.environ, {"API_KEY": "k"}):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/rerank",
            json={
                "model": "gemma4-reranker-v1",
                "query": "q",
                "documents": ["d"],
            },
            headers={"Authorization": "Bearer k"},
        )

    assert resp.status_code == 400
    assert "Audio duration" in resp.json()["error"]


def test_rerank_score_pair_unexpected_error(server):
    """Unexpected errors from _score_pair should return 400 with message."""
    server._score_pair = MagicMock(side_effect=RuntimeError("CUDA out of memory"))

    with patch.dict(os.environ, {"API_KEY": "k"}):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/rerank",
            json={
                "model": "gemma4-reranker-v1",
                "query": "q",
                "documents": ["d"],
            },
            headers={"Authorization": "Bearer k"},
        )

    assert resp.status_code == 400
    assert "Failed to score document 0" in resp.json()["error"]


# ---------------------------------------------------------------------------
# API versioning
# ---------------------------------------------------------------------------


def test_v1_rerank_endpoint(server):
    server._score_pair = MagicMock(return_value=0.8)

    with patch.dict(os.environ, {"API_KEY": "k"}):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/rerank",
            json={
                "model": "gemma4-reranker-v1",
                "query": "q",
                "documents": ["d"],
            },
            headers={"Authorization": "Bearer k"},
        )

    assert resp.status_code == 200


def test_v2_rerank_endpoint(server):
    server._score_pair = MagicMock(return_value=0.8)

    with patch.dict(os.environ, {"API_KEY": "k"}):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v2/rerank",
            json={
                "model": "gemma4-reranker-v1",
                "query": "q",
                "documents": ["d"],
            },
            headers={"Authorization": "Bearer k"},
        )

    assert resp.status_code == 200
    assert resp.json()["model"] == "gemma4-reranker-v1"


# ---------------------------------------------------------------------------
# Default model
# ---------------------------------------------------------------------------


def test_rerank_default_model(server):
    """When model field is omitted, should default to gemma4-reranker-v1."""
    server._score_pair = MagicMock(return_value=0.8)

    with patch.dict(os.environ, {"API_KEY": "k"}):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/rerank",
            json={"query": "q", "documents": ["d"]},
            headers={"Authorization": "Bearer k"},
        )

    assert resp.status_code == 200
    assert resp.json()["model"] == "gemma4-reranker-v1"
    call_args = server._score_pair.call_args
    assert call_args[0][0] == "gemma4-reranker-v1"  # model_name positional arg


# ---------------------------------------------------------------------------
# Multiple documents
# ---------------------------------------------------------------------------


def test_rerank_multiple_docs_all_scored(server):
    """Every document should be scored exactly once."""
    server._score_pair = MagicMock(side_effect=[0.1, 0.2, 0.3, 0.4, 0.5])

    with patch.dict(os.environ, {"API_KEY": "k"}):
        app = server.serve()
        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/rerank",
            json={
                "model": "gemma4-reranker-v1",
                "query": "q",
                "documents": ["d0", "d1", "d2", "d3", "d4"],
            },
            headers={"Authorization": "Bearer k"},
        )

    assert resp.status_code == 200
    assert server._score_pair.call_count == 5
    assert len(resp.json()["results"]) == 5
