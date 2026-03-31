"""Modal container image builders.

All images use uv_pip_install (instead of pip_install) for ~50% faster install speed.

Image types:
- transformers_image(): For models served via custom FastAPI (embedding, reranker, VL, OCR)
- transformers_tts_image(): For Qwen3-TTS text-to-speech (qwen-tts package)
- transformers_asr_image(): For Qwen3-ASR speech recognition (qwen-asr package)
- onnx_converter_image(): For ONNX conversion pipeline (CPU-only)

Models are loaded directly from HuggingFace Hub at container startup
via HF Xet protocol (~1GB/s). No R2 storage needed.
"""


import modal

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

PYTHON_VERSION = "3.13"


def transformers_image(*, flash_attn: bool = False) -> modal.Image:
    """Build a Modal image with transformers for custom FastAPI serving.

    Models are loaded from HuggingFace Hub via Xet protocol at container startup.
    HF_XET_HIGH_PERFORMANCE=1 enables high-speed download optimization (~1GB/s).

    Args:
        flash_attn: Install flash-attn package (needed for DeepSeek-OCR-2).
                    Requires CUDA devel image for nvcc compiler.
    """
    # DeepSeek-OCR-2 custom code imports LlamaFlashAttention2 (removed in transformers 4.46)
    tf_version = "transformers>=4.43,<4.46" if flash_attn else "transformers>=4.47"
    packages = [
        "torch>=2.4",
        "torchvision>=0.19",  # Required by Qwen3VLVideoProcessor (VL workers)
        tf_version,
        "safetensors>=0.4",
        "accelerate>=1.0",
        "huggingface_hub[hf_xet]",  # Fast model download via Xet protocol
        "fastapi>=0.115",
        "loguru>=0.7",
        "pydantic>=2.0",
        "pillow>=10.0",  # Required by Qwen2VLImageProcessor (VL workers)
    ]
    if flash_attn:
        # DeepSeek-OCR-2 custom modeling code dependencies
        packages.extend(["addict>=2.4", "einops>=0.8", "matplotlib>=3.8"])

    if flash_attn:
        # flash-attn pre-built wheel from GitHub releases (avoids 30+ min compile).
        # Pin torch==2.6.0 to match available wheel (torch 2.10 has no pre-built wheel).
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

    return img.env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",  # Required for GPU memory snapshot compatibility
        }
    ).add_local_python_source("ai_workers")


def transformers_tts_image() -> modal.Image:
    """Build a Modal image with qwen-tts for Qwen3-TTS text-to-speech.

    Models are loaded from HuggingFace Hub via Xet protocol at container startup.
    Includes soundfile and numpy for WAV audio output.
    """
    return (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .apt_install("libsndfile1", "sox")  # sox required by qwen-tts audio processing
        .uv_pip_install(
            "torch>=2.4",
            "transformers>=4.47",
            "safetensors>=0.4",
            "accelerate>=1.0",
            "huggingface_hub[hf_xet]",  # Fast model download via Xet protocol
            "fastapi>=0.115",
            "loguru>=0.7",
            "pydantic>=2.0",
            "qwen-tts",
            "soundfile>=0.12",
            "numpy>=2.0",
        )
        .env(
            {
                "HF_XET_HIGH_PERFORMANCE": "1",
                "TORCHINDUCTOR_COMPILE_THREADS": "1",  # Required for GPU memory snapshot compatibility
                "HF_HUB_CACHE": "/root/.cache/huggingface",  # Align with Volume mount + snapshot_download cache_dir
            }
        )
        .add_local_python_source("ai_workers")
    )


def transformers_asr_image() -> modal.Image:
    """Build a Modal image with qwen-asr for Qwen3-ASR speech recognition.

    Models are loaded from HuggingFace Hub via Xet protocol at container startup.
    Includes soundfile and python-multipart for audio file upload handling.
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
            "qwen-asr",
            "soundfile>=0.12",
            "numpy>=2.0",
            "python-multipart>=0.0.9",
        )
        .env(
            {
                "HF_XET_HIGH_PERFORMANCE": "1",
                "TORCHINDUCTOR_COMPILE_THREADS": "1",  # Required for GPU memory snapshot compatibility
                "HF_HUB_CACHE": "/root/.cache/huggingface",  # Align with Volume mount + snapshot_download cache_dir
            }
        )
        .add_local_python_source("ai_workers")
    )


def onnx_converter_image() -> modal.Image:
    """Build a Modal image for ONNX multi-variant conversion pipeline (CPU-only).

    Export HuggingFace models to ONNX + INT8 dynamic quantization + Q4F16
    (INT4 weights + FP16 activations). Used for Qwen3-Embedding / Reranker 0.6B
    (fastembed-compatible output).

    Dependencies:
    - onnxconverter-common: FP32 -> FP16 cast for Q4F16 variant
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
    """Build a Modal image for GGUF conversion pipeline (CPU-only).

    Convert HuggingFace models to GGUF format via llama.cpp.
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
            # Clone llama.cpp and build llama-quantize
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
