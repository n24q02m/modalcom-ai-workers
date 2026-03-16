"""Utility functions for AI workers."""

from __future__ import annotations

import base64
import io
import ipaddress
import socket
from urllib.parse import urlparse

import requests
from loguru import logger
from PIL import Image


def is_safe_url(url: str) -> bool:
    """Check if a URL is safe to fetch (prevents SSRF and LFI).

    Validates scheme and ensures the resolved IP is not private, loopback,
    link-local, or multicast.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            logger.warning("Rejected unsafe URL scheme: {}", parsed.scheme)
            return False

        if not parsed.hostname:
            return False

        # Resolve hostname to IP to prevent DNS rebinding/SSRF
        # getaddrinfo handles both IPv4 and IPv6
        addr_info = socket.getaddrinfo(parsed.hostname, None)

        for info in addr_info:
            ip_str = info[4][0]
            ip = ipaddress.ip_address(ip_str)

            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast:
                logger.warning(
                    "Rejected unsafe IP address: {} for hostname {}", ip_str, parsed.hostname
                )
                return False

        return True
    except Exception as e:
        logger.warning("URL validation failed for {}: {}", url, e)
        return False


def load_image_from_url(url: str) -> Image.Image:
    """Load image from URL or base64 data URI securely."""
    if url.startswith("data:image"):
        # Handle base64 data URI
        try:
            _header, b64_data = url.split(",", 1)
            image_bytes = base64.b64decode(b64_data)
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to load image from data URI: {e}") from e

    # Handle standard HTTP/HTTPS URLs securely
    if not is_safe_url(url):
        raise ValueError(f"Unsafe or invalid URL provided: {url}")

    try:
        # allow_redirects=False prevents SSRF via HTTP 302 redirects to internal IPs
        response = requests.get(url, stream=True, timeout=30, allow_redirects=False)
        if response.is_redirect:
            raise ValueError(f"Redirects are not allowed for security reasons: {url}")
        response.raise_for_status()
        return Image.open(response.raw).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to fetch image from URL {url}: {e}") from e
