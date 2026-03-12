"""Utility functions for ai-workers."""

import socket
import urllib.parse
from ipaddress import ip_address


def is_safe_url(url: str) -> bool:
    """Validate that a URL is safe to fetch (prevents SSRF/LFI).

    Ensures the scheme is http/https and the resolved IP address is not
    private, loopback, link-local, or multicast.
    """
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False

        hostname = parsed.hostname
        if not hostname:
            return False

        ip_addr = socket.gethostbyname(hostname)
        ip = ip_address(ip_addr)

        return not (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast)
    except Exception:
        return False
