"""Shared utilities for AI workers.

Provides SSRF-safe image loading for all vision workers (OCR, VL embedding, VL reranker).
"""

from __future__ import annotations

import base64
import contextlib
import io
import ipaddress
import socket
import threading
from typing import Any
from urllib.parse import urlparse

import requests
import urllib3.util.connection
from loguru import logger

# Global session for connection pooling to reduce latency
_session = requests.Session()

# Maximum image size: 20 MB
MAX_IMAGE_SIZE = 20 * 1024 * 1024
# Base64 encoding inflates size by ~4/3, so 20 MB decoded ~ 28 MB encoded
MAX_BASE64_SIZE = 28 * 1024 * 1024

# SSRF Protection: DNS Rebinding Mitigation
# We pin the hostname to the validated IP during the request.
_thread_local = threading.local()
_original_create_connection = urllib3.util.connection.create_connection


def _patched_create_connection(address: tuple[str, int], *args: Any, **kwargs: Any) -> Any:
    """Substituting hostname with pinned IP if available in current thread."""
    host, port = address
    pinned_ips = getattr(_thread_local, "pinned_ips", {})
    if host in pinned_ips:
        return _original_create_connection((pinned_ips[host], port), *args, **kwargs)
    return _original_create_connection(address, *args, **kwargs)


# Apply monkeypatch globally
urllib3.util.connection.create_connection = _patched_create_connection  # type: ignore


@contextlib.contextmanager
def _pin_hostname_to_ip(hostname: str, ip: str):
    """Context manager to pin a hostname to a specific IP for the current thread."""
    if not hasattr(_thread_local, "pinned_ips"):
        _thread_local.pinned_ips = {}

    previous_ip = _thread_local.pinned_ips.get(hostname)
    _thread_local.pinned_ips[hostname] = ip
    try:
        yield
    finally:
        if previous_ip is None:
            _thread_local.pinned_ips.pop(hostname, None)
        else:
            _thread_local.pinned_ips[hostname] = previous_ip


def is_safe_url(url: str) -> str | None:
    """Validate that a URL is safe to fetch (no SSRF).

    Checks:
    - Scheme must be http or https
    - Hostname must resolve to a public IP (not private/loopback/link-local/multicast)
    - Both IPv4 and IPv6 addresses are validated

    Args:
        url: The URL to validate.

    Returns:
        The first resolved safe IP (str) if the URL is safe, None otherwise.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        logger.warning("Failed to parse URL: {}", url)
        return None

    # Scheme check
    if parsed.scheme not in ("http", "https"):
        logger.warning("Rejected URL with disallowed scheme: {}", parsed.scheme)
        return None

    hostname = parsed.hostname
    if not hostname:
        logger.warning("Rejected URL with no hostname: {}", url)
        return None

    # Resolve hostname to IP addresses (both IPv4 and IPv6)
    try:
        addrinfos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror:
        logger.warning("DNS resolution failed for hostname: {}", hostname)
        return None

    if not addrinfos:
        logger.warning("No addresses resolved for hostname: {}", hostname)
        return None

    safe_ips = []
    for addrinfo in addrinfos:
        ip_str = addrinfo[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            logger.warning("Invalid IP address from DNS: {}", ip_str)
            return None

        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
            logger.warning(
                "Rejected URL {} — hostname {} resolves to non-public IP: {}", url, hostname, ip
            )
            return None
        safe_ips.append(ip_str)

    # Return the first one to use for pinning
    return safe_ips[0] if safe_ips else None


def load_image_from_url(url: str):
    """Load a PIL Image from a URL or base64 data URI with SSRF protection.

    Supports:
    - Base64 data URIs (data:image/...;base64,...) — no network call
    - HTTP/HTTPS URLs — validated against SSRF before fetching

    Args:
        url: Image URL or base64 data URI.

    Returns:
        PIL.Image.Image in RGB mode.

    Raises:
        ValueError: If the URL is blocked by SSRF protection or has an unsupported scheme.
        RuntimeError: If the image fetch or decode fails.
    """
    from PIL import Image

    # Handle base64 data URIs (no network call needed)
    if url.startswith("data:"):
        _header, b64_data = url.split(",", 1)
        if len(b64_data) > MAX_BASE64_SIZE:
            raise ValueError(
                f"Base64 image data exceeds size limit "
                f"({len(b64_data)} bytes encoded > {MAX_BASE64_SIZE} bytes max)"
            )
        try:
            image_bytes = base64.b64decode(b64_data)
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise RuntimeError("Failed to decode base64 image data URI") from e

    # SSRF check for HTTP URLs
    safe_ip = is_safe_url(url)
    if not safe_ip:
        raise ValueError(f"URL blocked by SSRF protection: {url}")

    hostname = urlparse(url).hostname
    if not hostname:
        raise ValueError(f"URL with no hostname: {url}")

    # Fetch image with safety controls and size limit
    # We pin the hostname to the validated IP to prevent DNS rebinding
    try:
        with _pin_hostname_to_ip(hostname, safe_ip):
            response = _session.get(
                url,
                allow_redirects=False,
                timeout=30,
                stream=True,
            )
            response.raise_for_status()

            # Read response in chunks with size limit to prevent memory exhaustion
            chunks: list[bytes] = []
            downloaded = 0
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                downloaded += len(chunk)
                if downloaded > MAX_IMAGE_SIZE:
                    response.close()
                    raise ValueError(
                        f"Image from URL exceeds size limit ({MAX_IMAGE_SIZE} bytes max): {url}"
                    )
                chunks.append(chunk)

            return Image.open(io.BytesIO(b"".join(chunks))).convert("RGB")
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to load image from URL: {url}") from e
