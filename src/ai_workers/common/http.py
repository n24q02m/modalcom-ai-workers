"""Utility functions for ai-workers related to HTTP and SSRF protection."""

import io
import socket
import urllib.parse
from ipaddress import ip_address

import httpx
from fastapi import HTTPException
from PIL import Image


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


async def load_image_async(url: str) -> Image.Image:
    """Safely fetch an image from a URL asynchronously.

    Includes SSRF validation and disables redirects.
    """
    if not is_safe_url(url):
        raise HTTPException(status_code=400, detail="Invalid or unsafe image URL.")

    try:
        async with httpx.AsyncClient(follow_redirects=False, timeout=30) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            image_bytes = resp.content
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except httpx.HTTPError as err:
        raise HTTPException(status_code=400, detail="Failed to fetch image.") from err
    except Exception as err:
        raise HTTPException(status_code=400, detail="Failed to process image.") from err


def load_image_sync(url: str) -> Image.Image:
    """Safely fetch an image from a URL synchronously.

    Includes SSRF validation and disables redirects.
    """
    if not is_safe_url(url):
        raise HTTPException(status_code=400, detail="Invalid or unsafe image URL.")

    try:
        with httpx.Client(follow_redirects=False, timeout=30) as client:
            resp = client.get(url)
            resp.raise_for_status()
            image_bytes = resp.content
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except httpx.HTTPError as err:
        raise HTTPException(status_code=400, detail="Failed to fetch image.") from err
    except Exception as err:
        raise HTTPException(status_code=400, detail="Failed to process image.") from err
