"""Tests for ASRServer FastAPI routes (Qwen3-ASR)."""

from __future__ import annotations

import io
import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from ai_workers.workers.asr import DEFAULT_MODEL, MODEL_CONFIGS, ASRServer


@pytest.fixture()
def server():
    return ASRServer()


def _make_audio_bytes() -> bytes:
    """Return minimal placeholder bytes to simulate an audio upload."""
    return b"RIFF....WAVEfmt "


def _client(server, api_key="k"):
    with patch.dict(os.environ, {"API_KEY": api_key}):
        import ai_workers.common.auth as auth_mod

        auth_mod._valid_keys = None
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
    data = resp.json()
    assert data["status"] == "ok"
    assert set(data["models"]) == set(MODEL_CONFIGS.keys())


def test_health_returns_model_names(server):
    app = server.serve()
    tc = TestClient(app)
    resp = tc.get("/health")
    assert "qwen3-asr-0.6b" in resp.json()["models"]
    assert "qwen3-asr-1.7b" in resp.json()["models"]


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
    server._load_audio = MagicMock(return_value=b"audio_data")
    server._transcribe = MagicMock(return_value="hello world")

    tc, key = _client(server)
    resp = tc.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", _make_audio_bytes(), "audio/wav")},
        data={"model": DEFAULT_MODEL, "response_format": "json"},
        headers={"Authorization": f"Bearer {key}"},
    )

    assert resp.status_code == 200
    assert resp.json()["text"] == "hello world"


def test_transcribe_text_format(server):
    server._load_audio = MagicMock(return_value=b"audio_data")
    server._transcribe = MagicMock(return_value="plain text")

    tc, key = _client(server)
    resp = tc.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", _make_audio_bytes(), "audio/wav")},
        data={"model": DEFAULT_MODEL, "response_format": "text"},
        headers={"Authorization": f"Bearer {key}"},
    )

    assert resp.status_code == 200
    assert resp.text == "plain text"


def test_transcribe_verbose_json(server):
    server._load_audio = MagicMock(return_value=b"audio_data")
    server._transcribe = MagicMock(return_value="hello")

    tc, key = _client(server)
    resp = tc.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", _make_audio_bytes(), "audio/wav")},
        data={"model": DEFAULT_MODEL, "response_format": "verbose_json"},
        headers={"Authorization": f"Bearer {key}"},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["task"] == "transcribe"
    assert data["text"] == "hello"
    assert data["language"] == "auto"
    assert data["duration"] == 0.0
    assert data["segments"] is None


def test_transcribe_with_language(server):
    server._load_audio = MagicMock(return_value=b"audio_data")
    mock_transcribe = MagicMock(return_value="bonjour")
    server._transcribe = mock_transcribe

    tc, key = _client(server)
    resp = tc.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", _make_audio_bytes(), "audio/wav")},
        data={"language": "fr"},
        headers={"Authorization": f"Bearer {key}"},
    )

    assert resp.status_code == 200
    # Verify language was passed to _transcribe
    call_kwargs = mock_transcribe.call_args
    assert call_kwargs[1]["language"] == "fr"


def test_transcribe_verbose_json_with_language(server):
    """Verbose JSON with explicit language should use that language."""
    server._load_audio = MagicMock(return_value=b"audio_data")
    server._transcribe = MagicMock(return_value="bonjour")

    tc, key = _client(server)
    resp = tc.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", _make_audio_bytes(), "audio/wav")},
        data={"language": "fr", "response_format": "verbose_json"},
        headers={"Authorization": f"Bearer {key}"},
    )

    assert resp.status_code == 200
    assert resp.json()["language"] == "fr"


# ---------------------------------------------------------------------------
# Model routing
# ---------------------------------------------------------------------------


def test_transcribe_light_model(server):
    server._load_audio = MagicMock(return_value=b"audio_data")
    mock_transcribe = MagicMock(return_value="light result")
    server._transcribe = mock_transcribe

    tc, key = _client(server)
    resp = tc.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", _make_audio_bytes(), "audio/wav")},
        data={"model": "qwen3-asr-0.6b"},
        headers={"Authorization": f"Bearer {key}"},
    )

    assert resp.status_code == 200
    assert mock_transcribe.call_args[0][0] == "qwen3-asr-0.6b"


def test_transcribe_heavy_model(server):
    server._load_audio = MagicMock(return_value=b"audio_data")
    mock_transcribe = MagicMock(return_value="heavy result")
    server._transcribe = mock_transcribe

    tc, key = _client(server)
    resp = tc.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", _make_audio_bytes(), "audio/wav")},
        data={"model": "qwen3-asr-1.7b"},
        headers={"Authorization": f"Bearer {key}"},
    )

    assert resp.status_code == 200
    assert mock_transcribe.call_args[0][0] == "qwen3-asr-1.7b"


def test_transcribe_unknown_model(server):
    tc, key = _client(server)
    resp = tc.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", _make_audio_bytes(), "audio/wav")},
        data={"model": "nonexistent-model"},
        headers={"Authorization": f"Bearer {key}"},
    )

    assert resp.status_code == 400
    assert "Unknown model" in resp.json()["error"]


# ---------------------------------------------------------------------------
# _transcribe method
# ---------------------------------------------------------------------------


def test_transcribe_method_string_result(server):
    """_transcribe should handle string result from model."""
    mock_model = MagicMock()
    mock_model.transcribe.return_value = " hello world "
    server.models = {DEFAULT_MODEL: mock_model}

    result = server._transcribe(DEFAULT_MODEL, b"audio_data")
    assert result == "hello world"


def test_transcribe_method_dict_result(server):
    """_transcribe should handle dict result with 'text' key."""
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"text": " bonjour "}
    server.models = {DEFAULT_MODEL: mock_model}

    result = server._transcribe(DEFAULT_MODEL, b"audio_data")
    assert result == "bonjour"


def test_transcribe_method_with_language(server):
    """_transcribe should pass language to model."""
    mock_model = MagicMock()
    mock_model.transcribe.return_value = "hola"
    server.models = {DEFAULT_MODEL: mock_model}

    server._transcribe(DEFAULT_MODEL, b"audio_data", language="es")
    mock_model.transcribe.assert_called_once_with(audio=b"audio_data", language="es")


def test_load_audio_returns_numpy_tuple(server):
    """_load_audio should convert bytes to (numpy_array, sample_rate) tuple."""
    import struct
    import wave

    # Generate minimal valid WAV file
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack("10h", *([0] * 10)))
    wav_bytes = buf.getvalue()

    result = server._load_audio(wav_bytes)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[1], int)  # sample rate
