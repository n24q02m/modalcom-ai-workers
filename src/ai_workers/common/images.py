"""Modal container image builders.

Two base images:
- vllm_image(): For models served via vLLM (text embeddings)
- transformers_image(): For models served via custom FastAPI (reranker, VL, OCR, ASR)
"""

from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

PYTHON_VERSION = "3.13"
MODELS_MOUNT_PATH = "/models"


def vllm_image() -> modal.Image:
    """Build a Modal image with vLLM for embedding serving.

    vLLM provides native OpenAI-compatible /v1/embeddings endpoint.
    """
    return (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .pip_install(
            "vllm>=0.8",
            "fastapi>=0.115",
            "loguru>=0.7",
        )
        .env({"HF_HUB_OFFLINE": "1"})  # No HF downloads at runtime
    )


def transformers_image(*, flash_attn: bool = False) -> modal.Image:
    """Build a Modal image with transformers for custom FastAPI serving.

    Args:
        flash_attn: Install flash-attn package (needed for DeepSeek-OCR-2).
    """
    packages = [
        "torch>=2.4",
        "transformers>=4.47",
        "safetensors>=0.4",
        "accelerate>=1.0",
        "fastapi>=0.115",
        "loguru>=0.7",
        "pydantic>=2.0",
    ]
    if flash_attn:
        packages.append("flash-attn>=2.6")

    return (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .pip_install(*packages)
        .env({"HF_HUB_OFFLINE": "1"})
    )


def transformers_audio_image() -> modal.Image:
    """Build a Modal image with transformers + audio processing for Whisper."""
    return (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .apt_install("ffmpeg", "libsndfile1")
        .pip_install(
            "torch>=2.4",
            "transformers>=4.47",
            "safetensors>=0.4",
            "accelerate>=1.0",
            "fastapi>=0.115",
            "loguru>=0.7",
            "pydantic>=2.0",
            "librosa>=0.10",
            "soundfile>=0.12",
            "python-multipart>=0.0.9",
        )
        .env({"HF_HUB_OFFLINE": "1"})
    )
