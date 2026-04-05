"""Shared utilities for AI workers.

Provides SSRF-safe image loading for all vision workers (OCR, VL embedding, VL reranker).
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import io
import ipaddress
import socket
import threading
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from collections.abc import Generator

import requests
from loguru import logger

# Global session for connection pooling to reduce latency
_session = requests.Session()

# Maximum image size: 20 MB

# SSRF IP Pinning using thread-local storage to prevent DNS rebinding
_thread_local = threading.local()


def _patched_create_connection(address, *args, **kwargs):
    """Monkeypatch for urllib3.util.connection.create_connection to force a pinned IP."""
    host, port = address
    pinned_ip = getattr(_thread_local, "pinned_ips", {}).get(host)
    if pinned_ip:
        address = (pinned_ip, port)

    return _original_create_connection(address, *args, **kwargs)


try:
    from urllib3.util import connection

    if not hasattr(connection, "_is_patched"):
        _original_create_connection = connection.create_connection
        connection.create_connection = _patched_create_connection  # type: ignore[assignment]
        connection._is_patched = True
except ImportError:
    logger.warning("Could not apply SSRF IP pinning: urllib3 not found")


@contextlib.contextmanager
def _pin_hostname_to_ip(hostname: str, ip: str) -> Generator[None]:
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


MAX_IMAGE_SIZE = 20 * 1024 * 1024
# Base64 encoding inflates size by ~4/3, so 20 MB decoded ~ 28 MB encoded
MAX_BASE64_SIZE = 28 * 1024 * 1024


def _get_safe_ips(url: str) -> list[str]:
    """Validate that a URL is safe to fetch and return its resolved IPs.

    Args:
        url: The URL to validate.

    Returns:
        List of safe IP addresses.

    Raises:
        ValueError: If the URL is unsafe or invalid.
    """
    try:
        parsed = urlparse(url)
    except Exception as e:
        raise ValueError(f"Failed to parse URL: {url}") from e

    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Rejected URL with disallowed scheme: {parsed.scheme}")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError(f"Rejected URL with no hostname: {url}")

    try:
        addrinfos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise ValueError(f"DNS resolution failed for hostname: {hostname}") from e

    if not addrinfos:
        raise ValueError(f"No addresses resolved for hostname: {hostname}")

    safe_ips = []
    for addrinfo in addrinfos:
        ip_str = addrinfo[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError as e:
            raise ValueError(f"Invalid IP address from DNS: {ip_str}") from e

        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
            raise ValueError(
                f"Rejected URL {url} — hostname {hostname} resolves to non-public IP: {ip}"
            )
        safe_ips.append(ip_str)

    return safe_ips


def is_safe_url(url: str) -> bool:
    """Validate that a URL is safe to fetch (no SSRF).

    Checks:
    - Scheme must be http or https
    - Hostname must resolve to a public IP (not private/loopback/link-local/multicast)
    - Both IPv4 and IPv6 addresses are validated

    Args:
        url: The URL to validate.

    Returns:
        True if the URL is safe to fetch, False otherwise.
    """
    try:
        return bool(_get_safe_ips(url))
    except ValueError as e:
        logger.warning(str(e))
        return False
    except Exception as e:
        logger.warning("Unexpected error validating URL {}: {}", url, e)
        return False


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
    from PIL import Image, UnidentifiedImageError

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
        except (binascii.Error, UnidentifiedImageError, OSError) as e:
            raise RuntimeError("Failed to decode base64 image data URI") from e

    # SSRF check for HTTP URLs and IP pinning
    try:
        safe_ips = _get_safe_ips(url)
    except ValueError as e:
        raise ValueError(f"URL blocked by SSRF protection: {url}") from e

    # Fetch image with safety controls and size limit
    try:
        parsed = urlparse(url)
        # Pin to the first resolved IP
        with _pin_hostname_to_ip(parsed.hostname, safe_ips[0]):
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


# ---------------------------------------------------------------------------
# HuggingFace Security
# ---------------------------------------------------------------------------

TRUSTED_ORGS = ["Qwen", "deepseek-ai"]


def validate_hf_repo_id(repo_id: str, trust_remote_code: bool) -> None:
    """Validate HuggingFace repository ID for security.

    Checks for:
    - Path traversal attempts (..)
    - Trusted organization if trust_remote_code is True

    Args:
        repo_id: HuggingFace repository ID (e.g. "org/model").
        trust_remote_code: Whether remote code execution is requested.

    Raises:
        ValueError: If the repository ID is invalid or untrusted.
    """
    if ".." in repo_id:
        raise ValueError(f"Invalid repository ID: path traversal detected in '{repo_id}'")

    if trust_remote_code:
        parts = repo_id.split("/")
        org = parts[0] if len(parts) > 1 else ""
        if org not in TRUSTED_ORGS:
            msg = (
                f"Untrusted organization '{org}' in '{repo_id}'. "
                f"trust_remote_code=True is only allowed for: {TRUSTED_ORGS}"
            )
            raise ValueError(msg)
