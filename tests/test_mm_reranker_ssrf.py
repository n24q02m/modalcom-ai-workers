import pytest
import socket
from unittest.mock import patch, MagicMock
from ai_workers.workers.mm_reranker import MmRerankerServer

@pytest.fixture
def server():
    return MmRerankerServer()

def test_load_image_ssrf_protection(server):
    """Verify that _load_image uses SSRF protection."""
    # Private IP address
    private_addrinfo = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("192.168.1.1", 0))]

    with patch("socket.getaddrinfo", return_value=private_addrinfo):
        with pytest.raises(ValueError, match="URL blocked by SSRF protection"):
            server._load_image("http://private.com/image.png")

def test_load_audio_ssrf_protection(server):
    """Verify that _load_audio uses SSRF protection."""
    private_addrinfo = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.1", 0))]

    with patch("socket.getaddrinfo", return_value=private_addrinfo):
        with pytest.raises(ValueError, match="URL blocked by SSRF protection"):
            server._load_audio("http://internal.com/audio.wav")

def test_load_video_ssrf_protection(server):
    """Verify that _load_video_frames uses SSRF protection."""
    private_addrinfo = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0))]

    with patch("socket.getaddrinfo", return_value=private_addrinfo):
        with pytest.raises(ValueError, match="URL blocked by SSRF protection"):
            server._load_video_frames("http://localhost/video.mp4")
