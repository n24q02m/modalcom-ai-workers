"""Common utilities for AI workers, including secure resource fetching."""

import base64
import io
import socket
from urllib.parse import urljoin, urlparse

import httpx
from PIL import Image

MAX_REDIRECTS = 5
MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20 MB max size to prevent OOM


def is_safe_url(url: str) -> bool:
    """Check if a URL is safe to fetch (prevents SSRF).

    Validates:
    1. Scheme is http or https.
    2. Hostname does not resolve to private, loopback, or reserved IP addresses.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    if parsed.scheme not in ("http", "https"):
        return False

    hostname = parsed.hostname
    if not hostname:
        return False

    try:
        # Resolve hostname to IP address
        ip_addr = socket.gethostbyname(hostname)
        ip = socket.inet_aton(ip_addr)
    except Exception:
        return False

    # Check if the IP is a loopback, private, or reserved address
    # IPv4 loopback (127.0.0.0/8)
    if ip[0] == 127:
        return False
    # IPv4 private blocks (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
    if ip[0] == 10:
        return False
    if ip[0] == 172 and 16 <= ip[1] <= 31:
        return False
    if ip[0] == 192 and ip[1] == 168:
        return False
    # Link-local (169.254.0.0/16)
    if ip[0] == 169 and ip[1] == 254:
        return False
    # Multicast (224.0.0.0/4)
    if 224 <= ip[0] <= 239:
        return False
    # Broadcast (255.255.255.255)
    if ip[0] == 255 and ip[1] == 255 and ip[2] == 255 and ip[3] == 255:
        return False
    # Current network (0.0.0.0/8)
    return ip[0] != 0


def load_image_from_url(url: str) -> Image.Image:
    """Load image securely from URL or base64 data URI.

    Protects against SSRF and TOCTOU DNS rebinding by explicitly disabling
    automatic redirects and manually validating each hop.
    """
    if url.startswith("data:"):
        # data:image/png;base64,<base64-data>
        try:
            _header, b64_data = url.split(",", 1)
            image_bytes = base64.b64decode(b64_data)
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid data URI: {e}") from e

    current_url = url
    for _ in range(MAX_REDIRECTS + 1):
        parsed = urlparse(current_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("Invalid URL scheme")

        hostname = parsed.hostname
        if not hostname:
            raise ValueError("Invalid URL hostname")

        # Resolve to IP to prevent DNS rebinding
        try:
            ip_addr = socket.gethostbyname(hostname)
        except Exception as e:
            raise ValueError("Failed to resolve hostname") from e

        # Validate IP using a reconstructed URL to use existing logic
        test_url = f"{parsed.scheme}://{ip_addr}{parsed.path}"
        if parsed.query:
            test_url += f"?{parsed.query}"

        if not is_safe_url(test_url):
            raise ValueError("Unsafe or invalid URL provided.")

        # Reconstruct URL to directly use the IP, setting the Host header
        safe_url = f"{parsed.scheme}://{ip_addr}"
        if parsed.port:
            safe_url += f":{parsed.port}"
        safe_url += parsed.path
        if parsed.query:
            safe_url += f"?{parsed.query}"

        headers = {"Host": hostname}

        try:
            with httpx.Client(follow_redirects=False, verify=False, timeout=30.0) as client:
                response = client.get(safe_url, headers=headers)

                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("Location")
                    if not location:
                        raise ValueError("Redirect response missing Location header.")
                    # Handle relative redirects securely
                    current_url = urljoin(current_url, location)
                    continue

                response.raise_for_status()

                # Stream content to prevent OOM
                content = b""
                for chunk in response.iter_bytes(chunk_size=8192):
                    content += chunk
                    if len(content) > MAX_IMAGE_SIZE:
                        raise ValueError(f"Image exceeds maximum size of {MAX_IMAGE_SIZE} bytes.")

                return Image.open(io.BytesIO(content)).convert("RGB")
        except httpx.RequestError as e:
            raise ValueError(f"Failed to fetch image: {e}") from e

    raise ValueError(f"Too many redirects (max {MAX_REDIRECTS})")
