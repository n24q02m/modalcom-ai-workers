import sys
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


# Mock modal to avoid import errors in test environment
class RecursiveMock(MagicMock):
    def __call__(self, *args, **kwargs):
        if any(
            name in self._mock_name if self._mock_name else ""
            for name in ("cls", "concurrent", "asgi_app", "enter", "function")
        ):
            return lambda x: x
        return super().__call__(*args, **kwargs)

    def __getattr__(self, name):
        if name in ("cls", "concurrent", "asgi_app", "enter", "function"):
            return lambda *args, **kwargs: lambda x: x
        return super().__getattr__(name)


sys.modules["modal"] = RecursiveMock()
sys.modules["ai_workers.common.images"] = MagicMock()

from ai_workers.workers.asr import ASRServer  # noqa: E402


def test_transcribe_file_too_large():
    server = ASRServer()
    server._load_audio = MagicMock()
    server._transcribe = MagicMock()

    # Mock auth
    with patch("ai_workers.common.auth.verify_api_key", side_effect=lambda r: None):
        app = server.serve()
        tc = TestClient(app)

        # 25MB + 1 byte
        max_size = 25 * 1024 * 1024
        large_content = b"a" * (max_size + 1024)

        resp = tc.post(
            "/v1/audio/transcriptions",
            files={"file": ("large.wav", large_content, "audio/wav")},
            data={"model": "qwen3-asr-0.6b"},
        )

        assert resp.status_code == 413
        assert "Audio file too large" in resp.json()["error"]
