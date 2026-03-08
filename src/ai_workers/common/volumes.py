"""Modal Volume for pre-downloaded model weights.

All workers share a single HuggingFace cache volume. Models are pre-downloaded
via the ``download_models`` function (run once), then loaded from the volume
at container startup instead of downloading from HF Hub each cold start.

Combined with GPU Memory Snapshots, this reduces cold start from >10 minutes
to ~5-10 seconds:
  1. Volume: model weights always on disk (no network download)
  2. GPU Snapshot: model already in GPU VRAM (no disk-to-GPU load)
"""

from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# Shared HuggingFace cache volume
# ---------------------------------------------------------------------------

HF_CACHE_DIR = "/root/.cache/huggingface"

hf_cache_vol = modal.Volume.from_name("ai-workers-hf-cache", create_if_missing=True)

# All model HF IDs — single source of truth for pre-download
ALL_MODEL_HF_IDS = [
    # Text Embedding (0.6B + 8B)
    "Qwen/Qwen3-Embedding-0.6B",
    "Qwen/Qwen3-Embedding-8B",
    # Text Reranker (0.6B + 8B)
    "Qwen/Qwen3-Reranker-0.6B",
    "Qwen/Qwen3-Reranker-8B",
    # VL Embedding (2B + 8B)
    "Qwen/Qwen3-VL-Embedding-2B",
    "Qwen/Qwen3-VL-Embedding-8B",
    # VL Reranker (2B + 8B)
    "Qwen/Qwen3-VL-Reranker-2B",
    "Qwen/Qwen3-VL-Reranker-8B",
    # OCR
    "deepseek-ai/DeepSeek-OCR-2",
    # ASR
    "openai/whisper-large-v3",
]

# ---------------------------------------------------------------------------
# Download function — run once to populate the volume
# ---------------------------------------------------------------------------

_download_image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install(
        "huggingface_hub[hf_xet]",
        "transformers>=4.47",
        "torch>=2.4",
        "loguru>=0.7",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

_download_app = modal.App("ai-workers-download")


@_download_app.function(
    image=_download_image,
    volumes={HF_CACHE_DIR: hf_cache_vol},
    timeout=3600,
)
def download_models() -> str:
    """Download all model weights to the shared HF cache volume.

    Usage:
        modal run src/ai_workers/common/volumes.py
    """
    from huggingface_hub import snapshot_download
    from loguru import logger

    targets = ALL_MODEL_HF_IDS
    results = []

    for hf_id in targets:
        logger.info("Downloading {} ...", hf_id)
        try:
            path = snapshot_download(
                hf_id,
                cache_dir=HF_CACHE_DIR,
                ignore_patterns=["*.gguf", "*.ot", "*.msgpack"],
            )
            logger.info("Downloaded {} to {}", hf_id, path)
            results.append(f"OK: {hf_id}")
        except Exception as e:
            logger.error("Failed to download {}: {}", hf_id, e)
            results.append(f"FAIL: {hf_id} ({e})")

    hf_cache_vol.commit()
    summary = "\n".join(results)
    logger.info("Download complete:\n{}", summary)
    return summary
