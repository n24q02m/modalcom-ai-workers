import pytest
from fastapi.testclient import TestClient
from ai_workers.workers.asr import ASRServer
import os
from unittest.mock import patch, MagicMock

def test_transcribe_file_too_large():
    server = ASRServer()
    server._load_audio = MagicMock()
    server._transcribe = MagicMock()

    with patch.dict(os.environ, {"API_KEY": "testkey"}):
        app = server.serve()

    tc = TestClient(app)

    # 25MB + 1 byte
    large_content = b"a" * (25 * 1024 * 1024 + 1)

    resp = tc.post(
        "/v1/audio/transcriptions",
        files={"file": ("large.wav", large_content, "audio/wav")},
        headers={"Authorization": "Bearer testkey"},
    )

    assert resp.status_code == 413
    assert "Audio file too large" in resp.json()["error"]

if __name__ == "__main__":
    test_transcribe_file_too_large()
    print("Test passed!")
