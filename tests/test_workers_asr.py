"""Tests for ASRServer FastAPI routes."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from ai_workers.workers.asr import MODEL_NAME, ASRServer


@pytest.fixture()
def server():
    return ASRServer()


def _make_audio_bytes() -> bytes:
    """Return minimal placeholder bytes to simulate an audio upload."""
    return b"RIFF....WAVEfmt "


def _client(server, api_key="k"):
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
    assert resp.json()["model"] == MODEL_NAME


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_transcribe_requires_auth(server):
    app = server.serve()
    tc = TestClient(app)
    resp = tc.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", _make_audio_bytes(), "audio/wav")},
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# /v1/audio/transcriptions — json format (default)
# ---------------------------------------------------------------------------


def test_transcribe_json_format(server):
    server._load_audio = MagicMock(return_value={"raw": [], "sampling_rate": 16000})
    server.pipe = MagicMock(return_value={"text": "hello world"})

    tc, key = _client(server)
    resp = tc.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", _make_audio_bytes(), "audio/wav")},
        data={"model": MODEL_NAME, "response_format": "json"},
        headers={"Authorization": f"Bearer {key}"},
    )

    assert resp.status_code == 200
    assert resp.json()["text"] == "hello world"


def test_transcribe_text_format(server):
    server._load_audio = MagicMock(return_value={"raw": [], "sampling_rate": 16000})
    server.pipe = MagicMock(return_value={"text": " plain text "})

    tc, key = _client(server)
    resp = tc.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", _make_audio_bytes(), "audio/wav")},
        data={"model": MODEL_NAME, "response_format": "text"},
        headers={"Authorization": f"Bearer {key}"},
    )

    assert resp.status_code == 200
    assert resp.text == "plain text"


def test_transcribe_verbose_json(server):
    pipe_result = {
        "text": "hello",
        "chunks": [
            {"timestamp": (0.0, 2.5), "text": "hello"},
        ],
    }
    server._load_audio = MagicMock(return_value={"raw": [], "sampling_rate": 16000})
    server.pipe = MagicMock(return_value=pipe_result)

    tc, key = _client(server)
    resp = tc.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", _make_audio_bytes(), "audio/wav")},
        data={"model": MODEL_NAME, "response_format": "verbose_json"},
        headers={"Authorization": f"Bearer {key}"},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["task"] == "transcribe"
    assert data["text"] == "hello"
    assert data["duration"] == pytest.approx(2.5)
    assert len(data["segments"]) == 1
    assert data["segments"][0]["start"] == pytest.approx(0.0)
    assert data["segments"][0]["end"] == pytest.approx(2.5)


def test_transcribe_verbose_json_empty_chunks(server):
    """Verbose JSON with no chunks: duration should be 0.0."""
    server._load_audio = MagicMock(return_value={"raw": [], "sampling_rate": 16000})
    server.pipe = MagicMock(return_value={"text": "hi", "chunks": []})

    tc, key = _client(server)
    resp = tc.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", _make_audio_bytes(), "audio/wav")},
        data={"response_format": "verbose_json"},
        headers={"Authorization": f"Bearer {key}"},
    )

    assert resp.status_code == 200
    assert resp.json()["duration"] == 0.0


def test_transcribe_with_language(server):
    server._load_audio = MagicMock(return_value={"raw": [], "sampling_rate": 16000})
    mock_pipe = MagicMock(return_value={"text": "bonjour"})
    server.pipe = mock_pipe

    tc, key = _client(server)
    resp = tc.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", _make_audio_bytes(), "audio/wav")},
        data={"language": "fr"},
        headers={"Authorization": f"Bearer {key}"},
    )

    assert resp.status_code == 200
    # Verify language was passed to pipe via generate_kwargs
    call_kwargs = mock_pipe.call_args[1]
    assert call_kwargs["generate_kwargs"]["language"] == "fr"


def test_transcribe_with_temperature(server):
    server._load_audio = MagicMock(return_value={"raw": [], "sampling_rate": 16000})
    mock_pipe = MagicMock(return_value={"text": "test"})
    server.pipe = mock_pipe

    tc, key = _client(server)
    resp = tc.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", _make_audio_bytes(), "audio/wav")},
        data={"temperature": "0.5"},
        headers={"Authorization": f"Bearer {key}"},
    )

    assert resp.status_code == 200
    call_kwargs = mock_pipe.call_args[1]
    assert call_kwargs["generate_kwargs"]["temperature"] == pytest.approx(0.5)
    assert call_kwargs["generate_kwargs"]["do_sample"] is True


def test_transcribe_verbose_json_none_timestamps(server):
    """Timestamps with None values should default to 0.0."""
    pipe_result = {
        "text": "hi",
        "chunks": [{"timestamp": (None, None), "text": "hi"}],
    }
    server._load_audio = MagicMock(return_value={"raw": [], "sampling_rate": 16000})
    server.pipe = MagicMock(return_value=pipe_result)

    tc, key = _client(server)
    resp = tc.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", _make_audio_bytes(), "audio/wav")},
        data={"response_format": "verbose_json"},
        headers={"Authorization": f"Bearer {key}"},
    )

    seg = resp.json()["segments"][0]
    assert seg["start"] == 0.0
    assert seg["end"] == 0.0
