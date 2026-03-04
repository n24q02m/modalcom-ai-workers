"""Modal container image builders.

Tất cả images dùng uv_pip_install (thay vì pip_install) cho tốc độ cài đặt nhanh hơn ~50%.

Hai loại image:
- transformers_image(): Cho models served qua custom FastAPI (embedding, reranker, VL, OCR, ASR)
- onnx_converter_image(): Cho ONNX conversion pipeline (CPU-only)

Models được tải trực tiếp từ HuggingFace Hub tại container startup
qua HF Xet protocol (~1GB/s). Không cần R2 storage.
"""

from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

PYTHON_VERSION = "3.13"


def transformers_image(*, flash_attn: bool = False) -> modal.Image:
    """Build a Modal image with transformers for custom FastAPI serving.

    Models được tải từ HuggingFace Hub qua Xet protocol tại container startup.
    HF_XET_HIGH_PERFORMANCE=1 bật tối ưu download tốc độ cao (~1GB/s).

    Args:
        flash_attn: Install flash-attn package (needed for DeepSeek-OCR-2).
                    Requires CUDA devel image for nvcc compiler.
    """
    # DeepSeek-OCR-2 custom code imports LlamaFlashAttention2 (removed in transformers 4.46)
    tf_version = "transformers>=4.43,<4.46" if flash_attn else "transformers>=4.47"
    packages = [
        "torch>=2.4",
        tf_version,
        "safetensors>=0.4",
        "accelerate>=1.0",
        "huggingface_hub[hf_xet]",  # Fast model download via Xet protocol
        "fastapi>=0.115",
        "loguru>=0.7",
        "pydantic>=2.0",
    ]
    if flash_attn:
        # DeepSeek-OCR-2 custom modeling code dependencies
        packages.extend(["addict>=2.4", "einops>=0.8", "matplotlib>=3.8"])

    if flash_attn:
        # flash-attn pre-built wheel từ GitHub releases (tránh compile 30+ phút).
        # Pin torch==2.6.0 để match wheel available (torch 2.10 không có pre-built wheel).
        # Wheel: cu12 + torch2.6 + Python 3.13 + cxx11abiFALSE + linux_x86_64.
        flash_attn_wheel = (
            "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
            "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp313-cp313-linux_x86_64.whl"
        )
        ocr_packages = [p for p in packages if not p.startswith("torch")]
        img = (
            modal.Image.debian_slim(python_version=PYTHON_VERSION)
            .uv_pip_install("torch==2.6.0", *ocr_packages)
            .uv_pip_install(flash_attn_wheel)
        )
    else:
        img = modal.Image.debian_slim(python_version=PYTHON_VERSION).uv_pip_install(*packages)

    return img.env({"HF_XET_HIGH_PERFORMANCE": "1"}).add_local_python_source("ai_workers")


def transformers_audio_image() -> modal.Image:
    """Build a Modal image with transformers + audio processing for Whisper.

    Models được tải từ HuggingFace Hub qua Xet protocol tại container startup.
    """
    return (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .apt_install("ffmpeg", "libsndfile1")
        .uv_pip_install(
            "torch>=2.4",
            "transformers>=4.47",
            "safetensors>=0.4",
            "accelerate>=1.0",
            "huggingface_hub[hf_xet]",  # Fast model download via Xet protocol
            "fastapi>=0.115",
            "loguru>=0.7",
            "pydantic>=2.0",
            "librosa>=0.10",
            "soundfile>=0.12",
            "python-multipart>=0.0.9",
        )
        .env({"HF_XET_HIGH_PERFORMANCE": "1"})
        .add_local_python_source("ai_workers")
    )


def onnx_converter_image() -> modal.Image:
    """Build a Modal image cho ONNX multi-variant conversion pipeline (CPU-only).

    Export HuggingFace models sang ONNX + INT8 dynamic quantization + Q4F16
    (INT4 weights + FP16 activations). Dung cho Qwen3-Embedding / Reranker 0.6B
    (fastembed-compatible output).

    Dependencies:
    - onnxconverter-common: FP32 -> FP16 cast cho Q4F16 variant
    - MatMulNBitsQuantizer (onnxruntime): INT4 weight quantization
    """
    return (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .uv_pip_install(
            "torch>=2.4",
            "transformers>=4.47",
            "safetensors>=0.4",
            "accelerate>=1.0",
            "huggingface_hub[hf_xet]",
            "onnx>=1.17",
            "onnxruntime>=1.21",
            "onnxscript>=0.1",
            "onnxconverter-common>=1.14",
            "loguru>=0.7",
        )
        .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    )


def gguf_converter_image() -> modal.Image:
    """Build a Modal image cho GGUF conversion pipeline (CPU-only).

    Convert HuggingFace models sang GGUF format via llama.cpp.
    Pipeline: HF model -> convert_hf_to_gguf.py (F16) -> llama-quantize (Q4_K_M).

    Image includes:
    - llama.cpp cloned + built from source (cmake, gcc)
    - Python dependencies for convert_hf_to_gguf.py (gguf, transformers, numpy, torch, sentencepiece)
    - HuggingFace Hub for model download/upload
    """
    return (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .apt_install("git", "cmake", "build-essential")
        .run_commands(
            # Clone llama.cpp va build llama-quantize
            "git clone --depth 1 https://github.com/ggml-org/llama.cpp /opt/llama.cpp",
            "cd /opt/llama.cpp && cmake -B build -DGGML_NATIVE=OFF && cmake --build build --target llama-quantize -j$(nproc)",
        )
        .uv_pip_install(
            "torch>=2.4",
            "transformers>=4.47",
            "safetensors>=0.4",
            "huggingface_hub[hf_xet]",
            "gguf>=0.6",
            "numpy>=2.0",
            "sentencepiece>=0.1",
            "loguru>=0.7",
        )
        .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    )
