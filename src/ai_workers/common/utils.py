import socket
from urllib.parse import urljoin, urlparse

import httpx
from loguru import logger


def is_safe_url(url: str) -> bool:
    """Check if URL is safe from SSRF.

    Allows http/https schemes only. Blocks private, loopback,
    and reserved IP ranges.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False

        if not parsed.hostname:
            return False

        ip = socket.gethostbyname(parsed.hostname)
        return not (
            ip.startswith("127.")
            or ip.startswith("10.")
            or ip.startswith("192.168.")
            or (ip.startswith("172.") and 16 <= int(ip.split(".")[1]) <= 31)
            or ip.startswith("169.254.")
            or ip.startswith("0.")
            or ip == "255.255.255.255"
        )
    except Exception as e:
        logger.warning(f"Error checking URL {url}: {e}")
        return False


def load_image_from_url(url: str, max_redirects: int = 3):
    """Load image from URL safely, mitigating SSRF and redirects."""
    import base64
    import io

    from PIL import Image

    if url.startswith("data:"):
        _header, b64_data = url.split(",", 1)
        image_bytes = base64.b64decode(b64_data)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    current_url = url
    redirects = 0

    with httpx.Client(follow_redirects=False, timeout=30) as client:
        while redirects <= max_redirects:
            if not is_safe_url(current_url):
                raise ValueError(f"Unsafe URL: {current_url}")

            resp = client.get(current_url)

            if resp.status_code in (301, 302, 303, 307, 308):
                location = resp.headers.get("Location")
                if not location:
                    raise ValueError("Redirect missing Location header")
                current_url = urljoin(current_url, location)
                redirects += 1
                continue

            resp.raise_for_status()
            image_bytes = resp.content
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")

        raise ValueError("Too many redirects")
