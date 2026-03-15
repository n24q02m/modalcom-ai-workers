from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from httpx import HTTPError
from PIL import Image

from ai_workers.common.http import is_safe_url, load_image_async, load_image_sync


def test_is_safe_url():
    # Test invalid schemes
    assert not is_safe_url("ftp://example.com/image.jpg")
    assert not is_safe_url("file:///etc/passwd")
    assert not is_safe_url("gopher://example.com")

    # Mock socket and ipaddress to simulate a public IP address
    with (
        patch("socket.gethostbyname", return_value="93.184.216.34"),
        patch("ai_workers.common.http.ip_address") as mock_ip,
    ):
        mock_ip.return_value.is_private = False
        mock_ip.return_value.is_loopback = False
        mock_ip.return_value.is_link_local = False
        mock_ip.return_value.is_multicast = False

        assert is_safe_url("https://example.com/image.jpg")
        assert is_safe_url("http://example.com/image.jpg")

    # Mock private/loopback IPs
    with (
        patch("socket.gethostbyname", return_value="127.0.0.1"),
        patch("ai_workers.common.http.ip_address") as mock_ip,
    ):
        mock_ip.return_value.is_private = False
        mock_ip.return_value.is_loopback = True
        mock_ip.return_value.is_link_local = False
        mock_ip.return_value.is_multicast = False

        assert not is_safe_url("http://localhost/image.jpg")


def test_is_safe_url_exception():
    with patch("socket.gethostbyname", side_effect=Exception("DNS resolution failed")):
        assert not is_safe_url("https://invalid-domain.local/image.jpg")


@pytest.mark.asyncio
async def test_load_image_async_success():
    with patch("ai_workers.common.http.is_safe_url", return_value=True):
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.content = b"fake_image_bytes"
        mock_client.get.return_value = mock_resp

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("PIL.Image.open") as mock_open,
        ):
            mock_image = MagicMock(spec=Image.Image)
            mock_open.return_value.convert.return_value = mock_image

            result = await load_image_async("https://example.com/image.png")
            assert result == mock_image
            mock_client.get.assert_called_once_with("https://example.com/image.png")


@pytest.mark.asyncio
async def test_load_image_async_unsafe_url():
    with patch("ai_workers.common.http.is_safe_url", return_value=False):
        with pytest.raises(HTTPException) as exc_info:
            await load_image_async("file:///etc/passwd")
        assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_load_image_async_http_error():
    with patch("ai_workers.common.http.is_safe_url", return_value=True):
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.get.side_effect = HTTPError("Not found")

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(HTTPException) as exc_info:
                await load_image_async("https://example.com/image.png")
            assert exc_info.value.status_code == 400
            assert "Failed to fetch image" in exc_info.value.detail


def test_load_image_sync_success():
    with patch("ai_workers.common.http.is_safe_url", return_value=True):
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.content = b"fake_image_bytes"
        mock_client.get.return_value = mock_resp

        with patch("httpx.Client", return_value=mock_client), patch("PIL.Image.open") as mock_open:
            mock_image = MagicMock(spec=Image.Image)
            mock_open.return_value.convert.return_value = mock_image

            result = load_image_sync("https://example.com/image.png")
            assert result == mock_image
            mock_client.get.assert_called_once_with("https://example.com/image.png")


def test_load_image_sync_unsafe_url():
    with patch("ai_workers.common.http.is_safe_url", return_value=False):
        with pytest.raises(HTTPException) as exc_info:
            load_image_sync("file:///etc/passwd")
        assert exc_info.value.status_code == 400
