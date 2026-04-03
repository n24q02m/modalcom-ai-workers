"""Tests for ai_workers.common.utils (SSRF protection + image loading)."""

from __future__ import annotations

import base64
import io
import socket
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from ai_workers.common.utils import is_safe_url, load_image_from_url

# ---------------------------------------------------------------------------
# is_safe_url
# ---------------------------------------------------------------------------

# Public IPs that should be allowed
_PUBLIC_ADDRINFO = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]

# Private/loopback/link-local/multicast addresses
_PRIVATE_ADDRINFO = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("192.168.1.1", 0))]
_LOOPBACK_ADDRINFO = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0))]
_LINK_LOCAL_ADDRINFO = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("169.254.1.1", 0))]
_MULTICAST_ADDRINFO = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("224.0.0.1", 0))]
_IPV6_LOOPBACK_ADDRINFO = [(socket.AF_INET6, socket.SOCK_STREAM, 0, "", ("::1", 0, 0, 0))]
_IPV6_PRIVATE_ADDRINFO = [(socket.AF_INET6, socket.SOCK_STREAM, 0, "", ("fd00::1", 0, 0, 0))]
_RESERVED_ADDRINFO = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("0.0.0.0", 0))]


class TestIsSafeUrl:
    """Tests for is_safe_url."""

    def test_public_http_url(self):
        with patch("socket.getaddrinfo", return_value=_PUBLIC_ADDRINFO):
            assert is_safe_url("http://example.com/image.png") is True

    def test_public_https_url(self):
        with patch("socket.getaddrinfo", return_value=_PUBLIC_ADDRINFO):
            assert is_safe_url("https://example.com/image.png") is True

    def test_rejects_ftp_scheme(self):
        assert is_safe_url("ftp://example.com/file") is False

    def test_rejects_file_scheme(self):
        assert is_safe_url("file:///etc/passwd") is False

    def test_rejects_javascript_scheme(self):
        assert is_safe_url("javascript:alert(1)") is False

    def test_rejects_empty_scheme(self):
        assert is_safe_url("://example.com") is False

    def test_rejects_no_hostname(self):
        assert is_safe_url("http://") is False

    def test_rejects_private_ip(self):
        with patch("socket.getaddrinfo", return_value=_PRIVATE_ADDRINFO):
            assert is_safe_url("http://192.168.1.1/image.png") is False

    def test_rejects_loopback(self):
        with patch("socket.getaddrinfo", return_value=_LOOPBACK_ADDRINFO):
            assert is_safe_url("http://localhost/image.png") is False

    def test_rejects_link_local(self):
        with patch("socket.getaddrinfo", return_value=_LINK_LOCAL_ADDRINFO):
            assert is_safe_url("http://169.254.1.1/image.png") is False

    def test_rejects_multicast(self):
        with patch("socket.getaddrinfo", return_value=_MULTICAST_ADDRINFO):
            assert is_safe_url("http://224.0.0.1/image.png") is False

    def test_rejects_reserved(self):
        with patch("socket.getaddrinfo", return_value=_RESERVED_ADDRINFO):
            assert is_safe_url("http://0.0.0.0/image.png") is False

    def test_rejects_ipv6_loopback(self):
        with patch("socket.getaddrinfo", return_value=_IPV6_LOOPBACK_ADDRINFO):
            assert is_safe_url("http://[::1]/image.png") is False

    def test_rejects_ipv6_private(self):
        with patch("socket.getaddrinfo", return_value=_IPV6_PRIVATE_ADDRINFO):
            assert is_safe_url("http://[fd00::1]/image.png") is False

    def test_rejects_dns_failure(self):
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("DNS failed")):
            assert is_safe_url("http://nonexistent.invalid/image.png") is False

    def test_rejects_empty_resolution(self):
        with patch("socket.getaddrinfo", return_value=[]):
            assert is_safe_url("http://example.com/image.png") is False

    def test_rejects_invalid_ip_format(self):
        invalid_addrinfo = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("not-an-ip", 0))]
        with patch("socket.getaddrinfo", return_value=invalid_addrinfo):
            assert is_safe_url("http://example.com/image.png") is False

    def test_mixed_addresses_one_private(self):
        """If any resolved address is private, reject the URL."""
        mixed = _PUBLIC_ADDRINFO + _PRIVATE_ADDRINFO
        with patch("socket.getaddrinfo", return_value=mixed):
            assert is_safe_url("http://example.com/image.png") is False

    def test_url_with_port(self):
        with patch("socket.getaddrinfo", return_value=_PUBLIC_ADDRINFO):
            assert is_safe_url("https://example.com:8080/image.png") is True

    def test_url_with_path_and_query(self):
        with patch("socket.getaddrinfo", return_value=_PUBLIC_ADDRINFO):
            assert is_safe_url("https://example.com/path?q=1&b=2") is True

    def test_rejects_10_x_private(self):
        addrinfo = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.1", 0))]
        with patch("socket.getaddrinfo", return_value=addrinfo):
            assert is_safe_url("http://10.0.0.1/image.png") is False

    def test_rejects_172_16_private(self):
        addrinfo = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("172.16.0.1", 0))]
        with patch("socket.getaddrinfo", return_value=addrinfo):
            assert is_safe_url("http://172.16.0.1/image.png") is False


# ---------------------------------------------------------------------------
# load_image_from_url
# ---------------------------------------------------------------------------


def _make_test_image_bytes(size=(2, 2), color=(0, 255, 0)) -> bytes:
    """Create a small PNG image in memory and return its bytes."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestLoadImageFromUrl:
    """Tests for load_image_from_url."""

    def test_base64_data_uri(self):
        img_bytes = _make_test_image_bytes(size=(1, 1), color=(255, 0, 0))
        b64 = base64.b64encode(img_bytes).decode()
        data_uri = f"data:image/png;base64,{b64}"

        result = load_image_from_url(data_uri)
        assert result.mode == "RGB"
        assert result.size == (1, 1)

    def test_base64_data_uri_jpeg(self):
        img = Image.new("RGB", (1, 1), color=(0, 0, 255))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        data_uri = f"data:image/jpeg;base64,{b64}"

        result = load_image_from_url(data_uri)
        assert result.mode == "RGB"

    def test_base64_invalid_data_raises(self):
        with pytest.raises(RuntimeError, match="Failed to decode base64 image data URI"):
            load_image_from_url("data:image/png;base64,!!!invalid!!!")

    def test_http_url_success(self):
        img_bytes = _make_test_image_bytes()
        mock_resp = MagicMock()
        mock_resp.iter_content = MagicMock(return_value=iter([img_bytes]))
        mock_resp.raise_for_status = MagicMock()

        with (
            patch("ai_workers.common.utils.is_safe_url", return_value=True),
            patch("ai_workers.common.utils._session.get", return_value=mock_resp) as mock_get,
        ):
            result = load_image_from_url("https://example.com/image.png")

        assert result.mode == "RGB"
        assert result.size == (2, 2)
        mock_get.assert_called_once_with(
            "https://example.com/image.png",
            allow_redirects=False,
            timeout=30,
            stream=True,
        )

    def test_http_url_ssrf_blocked(self):
        with (
            patch("ai_workers.common.utils.is_safe_url", return_value=False),
            pytest.raises(ValueError, match="URL blocked by SSRF protection"),
        ):
            load_image_from_url("http://192.168.1.1/image.png")

    def test_http_url_fetch_failure(self):
        with (
            patch("ai_workers.common.utils.is_safe_url", return_value=True),
            patch(
                "ai_workers.common.utils._session.get",
                side_effect=Exception("Connection refused"),
            ),
            pytest.raises(RuntimeError, match="Failed to load image from URL"),
        ):
            load_image_from_url("https://example.com/unreachable.png")

    def test_http_url_no_redirects(self):
        """Verify allow_redirects=False is passed to prevent redirect-based SSRF."""
        img_bytes = _make_test_image_bytes()
        mock_resp = MagicMock()
        mock_resp.iter_content = MagicMock(return_value=iter([img_bytes]))
        mock_resp.raise_for_status = MagicMock()

        with (
            patch("ai_workers.common.utils.is_safe_url", return_value=True),
            patch("ai_workers.common.utils._session.get", return_value=mock_resp) as mock_get,
        ):
            load_image_from_url("https://example.com/image.png")

        call_kwargs = mock_get.call_args.kwargs
        assert call_kwargs["allow_redirects"] is False
        assert call_kwargs["timeout"] == 30

    def test_converts_to_rgb(self):
        """Ensure RGBA/P/L images are converted to RGB."""
        img = Image.new("RGBA", (1, 1), color=(255, 0, 0, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        data_uri = f"data:image/png;base64,{b64}"

        result = load_image_from_url(data_uri)
        assert result.mode == "RGB"
