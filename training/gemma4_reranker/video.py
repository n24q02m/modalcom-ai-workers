"""Video frame extraction utilities for training and serving.

Uses PyAV (av) for efficient video decoding. Extracts uniformly-spaced
frames at ~1 FPS to provide visual context for video reranking.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import av
import numpy as np

if TYPE_CHECKING:
    from PIL import Image


def extract_frames(
    video_path: str | Path,
    max_frames: int = 4,
    max_duration_s: float = 30.0,
) -> list[Image.Image]:
    """Extract uniformly-spaced frames from a video file.

    Args:
        video_path: Path to the video file on disk.
        max_frames: Maximum number of frames to extract (~1 FPS).
        max_duration_s: Maximum allowed video duration in seconds.

    Returns:
        List of PIL.Image (RGB) frames.

    Raises:
        ValueError: If video exceeds max_duration_s.
        FileNotFoundError: If video_path does not exist.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base)

        if duration > max_duration_s:
            raise ValueError(
                f"Video duration {duration:.1f}s exceeds maximum {max_duration_s}s"
            )

        n_frames = min(max_frames, max(1, int(duration)))  # ~1 FPS
        timestamps = np.linspace(0, duration, n_frames, endpoint=False)

        frames: list[Image.Image] = []
        for ts in timestamps:
            target_pts = int(ts / stream.time_base)
            container.seek(target_pts, stream=stream)
            for frame in container.decode(video=0):
                frames.append(frame.to_image().convert("RGB"))
                break

    finally:
        container.close()

    return frames[:max_frames]


def extract_frames_from_url(
    url: str,
    max_frames: int = 4,
    max_duration_s: float = 30.0,
    timeout_s: int = 30,
) -> list[Image.Image]:
    """Download a video from URL and extract frames.

    Args:
        url: HTTP(S) URL to the video file.
        max_frames: Maximum number of frames to extract.
        max_duration_s: Maximum allowed video duration.
        timeout_s: Download timeout in seconds.

    Returns:
        List of PIL.Image (RGB) frames.
    """
    import requests as http_requests

    resp = http_requests.get(url, timeout=timeout_s)
    resp.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        tmp.write(resp.content)
        tmp.flush()
        return extract_frames(tmp.name, max_frames=max_frames, max_duration_s=max_duration_s)
