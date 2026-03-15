"""Tests for OCRServer FastAPI routes."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from ai_workers.workers.ocr import MODEL_NAME, OCRServer


@pytest.fixture()
def server():
    return OCRServer()


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


def test_chat_completions_requires_auth(server):
    app = server.serve()
    tc = TestClient(app)
    resp = tc.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "extract text"}],
        },
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# No image provided
# ---------------------------------------------------------------------------


def test_chat_completions_no_image(server):
    tc, key = _client(server)
    resp = tc.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "extract text"}],
        },
        headers={"Authorization": f"Bearer {key}"},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert "No image" in data["choices"][0]["message"]["content"]
    assert data["choices"][0]["finish_reason"] == "stop"


# ---------------------------------------------------------------------------
# With image — list content format
# ---------------------------------------------------------------------------


def test_chat_completions_with_image_url(server):
    fake_image = MagicMock()
    server._load_image_from_url = AsyncMock(return_value=fake_image)
    server._run_ocr = MagicMock(return_value="Extracted text from image")

    tc, key = _client(server)
    resp = tc.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all text"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                    ],
                }
            ],
        },
        headers={"Authorization": f"Bearer {key}"},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "Extracted text from image"
    assert data["choices"][0]["message"]["role"] == "assistant"
    server._load_image_from_url.assert_called_once_with("https://example.com/img.png")
    server._run_ocr.assert_called_once_with(fake_image, "Extract all text")


def test_chat_completions_response_has_id(server):
    server._load_image_from_url = AsyncMock(return_value=MagicMock())
    server._run_ocr = MagicMock(return_value="text")

    tc, key = _client(server)
    resp = tc.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                    ],
                }
            ],
        },
        headers={"Authorization": f"Bearer {key}"},
    )

    data = resp.json()
    assert data["id"].startswith("chatcmpl-")
    assert data["object"] == "chat.completion"


# ---------------------------------------------------------------------------
# _process_image_content unit tests
# ---------------------------------------------------------------------------


def test_process_image_content_text_only(server):
    content = [{"type": "text", "text": "hello"}]
    text, url = server._process_image_content(content)
    assert text == "hello"
    assert url is None


def test_process_image_content_with_image(server):
    content = [
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": "https://img.url/x.png"}},
    ]
    text, url = server._process_image_content(content)
    assert text == "describe"
    assert url == "https://img.url/x.png"


def test_process_image_content_image_only(server):
    content = [{"type": "image_url", "image_url": {"url": "http://a.b/c.jpg"}}]
    text, url = server._process_image_content(content)
    assert text == ""
    assert url == "http://a.b/c.jpg"


# ---------------------------------------------------------------------------
# _load_image_from_url unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_image_from_url_base64(server):
    """base64 data URI should be decoded without network call."""
    import base64

    from PIL import Image

    # Create a tiny 1x1 red PNG
    img = Image.new("RGB", (1, 1), color=(255, 0, 0))
    buf = __import__("io").BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    data_uri = f"data:image/png;base64,{b64}"

    result = await server._load_image_from_url(data_uri)
    assert result.mode == "RGB"
    assert result.size == (1, 1)


@pytest.mark.asyncio
async def test_load_image_from_url_network(server):
    """Regular URL should use urllib.request.urlopen."""
    import io

    from PIL import Image

    img = Image.new("RGB", (2, 2), color=(0, 255, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.content = buf.getvalue()
    mock_client.get.return_value = mock_resp
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await server._load_image_from_url("https://example.com/img.png")

    assert result.mode == "RGB"
