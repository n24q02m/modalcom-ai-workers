"""Tests for TTSServer FastAPI routes (Qwen3-TTS CustomVoice)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from ai_workers.workers.tts import (
    DEFAULT_MODEL,
    DEFAULT_VOICE,
    MODEL_CONFIGS,
    SUPPORTED_SPEAKERS,
    TTSServer,
)


@pytest.fixture()
def server():
    return TTSServer()


@pytest.fixture()
def client(server):
    app = server.serve()
    return TestClient(app, raise_server_exceptions=True)


def _client(server, api_key="k"):
    with patch.dict(os.environ, {"WORKER_API_KEY": api_key}):
        app = server.serve()
    return TestClient(app, raise_server_exceptions=True), api_key


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


def test_health_no_auth(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert set(data["models"]) == set(MODEL_CONFIGS.keys())
    assert "speakers" in data
    assert "vivian" in data["speakers"]


def test_health_returns_model_names(client):
    resp = client.get("/health")
    assert "qwen3-tts-0.6b" in resp.json()["models"]
    assert "qwen3-tts-1.7b" in resp.json()["models"]


# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------


def test_speech_requires_auth(server):
    with patch.dict(os.environ, {"WORKER_API_KEY": "secret"}):
        app = server.serve()
    tc = TestClient(app)
    resp = tc.post(
        "/v1/audio/speech",
        json={"model": "qwen3-tts-0.6b", "input": "hello"},
    )
    assert resp.status_code == 401


def test_speech_with_valid_key(server):
    import numpy as np

    fake_wav = (np.zeros(100, dtype=np.float32), 24000)
    server._synthesize = MagicMock(return_value=fake_wav)

    tc, key = _client(server)
    resp = tc.post(
        "/v1/audio/speech",
        json={"model": "qwen3-tts-0.6b", "input": "hello"},
        headers={"Authorization": f"Bearer {key}"},
    )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /v1/audio/speech — invalid model / voice
# ---------------------------------------------------------------------------


def test_speech_unknown_model(server):
    tc, key = _client(server)
    resp = tc.post(
        "/v1/audio/speech",
        json={"model": "nonexistent-model", "input": "hello"},
        headers={"Authorization": f"Bearer {key}"},
    )
    assert resp.status_code == 400
    assert "Unknown model" in resp.json()["error"]


def test_speech_unknown_voice(server):
    tc, key = _client(server)
    resp = tc.post(
        "/v1/audio/speech",
        json={"model": "qwen3-tts-0.6b", "input": "hello", "voice": "nonexistent"},
        headers={"Authorization": f"Bearer {key}"},
    )
    assert resp.status_code == 400
    assert "Unknown voice" in resp.json()["error"]


# ---------------------------------------------------------------------------
# /v1/audio/speech — valid requests
# ---------------------------------------------------------------------------


def test_speech_returns_wav_audio(server):
    import numpy as np

    fake_wav = np.zeros(24000, dtype=np.float32)
    server._synthesize = MagicMock(return_value=(fake_wav, 24000))

    tc, key = _client(server)
    resp = tc.post(
        "/v1/audio/speech",
        json={"model": "qwen3-tts-0.6b", "input": "Hello world"},
        headers={"Authorization": f"Bearer {key}"},
    )

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"


def test_speech_default_model(server):
    import numpy as np

    fake_wav = np.zeros(100, dtype=np.float32)
    server._synthesize = MagicMock(return_value=(fake_wav, 24000))

    tc, key = _client(server)
    resp = tc.post(
        "/v1/audio/speech",
        json={"input": "test"},
        headers={"Authorization": f"Bearer {key}"},
    )

    assert resp.status_code == 200
    call_args = server._synthesize.call_args
    assert call_args[0][0] == DEFAULT_MODEL


def test_speech_heavy_model(server):
    import numpy as np

    fake_wav = np.zeros(100, dtype=np.float32)
    server._synthesize = MagicMock(return_value=(fake_wav, 24000))

    tc, key = _client(server)
    resp = tc.post(
        "/v1/audio/speech",
        json={"model": "qwen3-tts-1.7b", "input": "test"},
        headers={"Authorization": f"Bearer {key}"},
    )

    assert resp.status_code == 200
    call_args = server._synthesize.call_args
    assert call_args[0][0] == "qwen3-tts-1.7b"


def test_speech_with_voice(server):
    import numpy as np

    fake_wav = np.zeros(100, dtype=np.float32)
    server._synthesize = MagicMock(return_value=(fake_wav, 24000))

    tc, key = _client(server)
    resp = tc.post(
        "/v1/audio/speech",
        json={"model": "qwen3-tts-0.6b", "input": "Hello", "voice": "ryan"},
        headers={"Authorization": f"Bearer {key}"},
    )

    assert resp.status_code == 200


def test_speech_with_language(server):
    import numpy as np

    fake_wav = np.zeros(100, dtype=np.float32)
    server._synthesize = MagicMock(return_value=(fake_wav, 24000))

    tc, key = _client(server)
    resp = tc.post(
        "/v1/audio/speech",
        json={"model": "qwen3-tts-0.6b", "input": "Bonjour", "language": "French"},
        headers={"Authorization": f"Bearer {key}"},
    )

    assert resp.status_code == 200


def test_speech_with_instruct(server):
    import numpy as np

    fake_wav = np.zeros(100, dtype=np.float32)
    server._synthesize = MagicMock(return_value=(fake_wav, 24000))

    tc, key = _client(server)
    resp = tc.post(
        "/v1/audio/speech",
        json={
            "model": "qwen3-tts-0.6b",
            "input": "I am so happy!",
            "voice": "vivian",
            "instruct": "Very happy and excited",
        },
        headers={"Authorization": f"Bearer {key}"},
    )

    assert resp.status_code == 200


def test_speech_content_disposition_header(server):
    import numpy as np

    fake_wav = np.zeros(100, dtype=np.float32)
    server._synthesize = MagicMock(return_value=(fake_wav, 24000))

    tc, key = _client(server)
    resp = tc.post(
        "/v1/audio/speech",
        json={"model": "qwen3-tts-0.6b", "input": "test"},
        headers={"Authorization": f"Bearer {key}"},
    )

    assert resp.status_code == 200
    assert "attachment" in resp.headers.get("content-disposition", "")


# ---------------------------------------------------------------------------
# _synthesize method
# ---------------------------------------------------------------------------


def test_synthesize_default_speaker(server):
    """_synthesize should use generate_custom_voice with default speaker."""
    import numpy as np

    mock_model = MagicMock()
    mock_model.generate_custom_voice.return_value = ([np.zeros(100, dtype=np.float32)], 24000)
    server.models = {"qwen3-tts-0.6b": mock_model}

    _wavs, sample_rate = server._synthesize("qwen3-tts-0.6b", "hello", "vivian", "English")

    mock_model.generate_custom_voice.assert_called_once_with(
        text="hello", language="English", speaker="vivian"
    )
    assert sample_rate == 24000


def test_synthesize_with_instruct(server):
    """_synthesize with instruct should pass it to generate_custom_voice."""
    import numpy as np

    mock_model = MagicMock()
    mock_model.generate_custom_voice.return_value = ([np.zeros(100, dtype=np.float32)], 24000)
    server.models = {"qwen3-tts-0.6b": mock_model}

    server._synthesize("qwen3-tts-0.6b", "hello", "ryan", "English", instruct="Angry tone")

    mock_model.generate_custom_voice.assert_called_once_with(
        text="hello", language="English", speaker="ryan", instruct="Angry tone"
    )


def test_synthesize_different_speaker(server):
    """_synthesize should pass the chosen speaker to generate_custom_voice."""
    import numpy as np

    mock_model = MagicMock()
    mock_model.generate_custom_voice.return_value = ([np.zeros(100, dtype=np.float32)], 24000)
    server.models = {"qwen3-tts-0.6b": mock_model}

    server._synthesize("qwen3-tts-0.6b", "test", "aiden", "Auto")

    call_kwargs = mock_model.generate_custom_voice.call_args[1]
    assert call_kwargs["speaker"] == "aiden"


def test_default_voice_constant():
    assert DEFAULT_VOICE == "vivian"
    assert DEFAULT_VOICE in SUPPORTED_SPEAKERS
