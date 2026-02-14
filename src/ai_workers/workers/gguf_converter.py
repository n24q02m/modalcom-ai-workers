"""Modal CPU function de convert HuggingFace models sang GGUF Q4_K_M.

Chay tren Modal CPU container (32GB RAM) thay vi may local.
Download tu HuggingFace Hub -> convert_hf_to_gguf.py (F16) -> llama-quantize (Q4_K_M)
-> push HF Hub.

Output structure:
  {hf_repo_id}/
    gguf/{model_name}-q4_k_m.gguf   # GGUF Q4_K_M quantized

Flow:
  CLI (local) -> gguf_convert_model.remote() -> Modal CPU container:
    1. Download model tu HuggingFace Hub (safetensors)
    2. Run convert_hf_to_gguf.py -> F16 GGUF (intermediate)
    3. Run llama-quantize -> Q4_K_M GGUF
    4. Push len HuggingFace Hub (existing repo)
"""

from __future__ import annotations

from dataclasses import dataclass

import modal

from ai_workers.common.images import gguf_converter_image

# ---------------------------------------------------------------------------
# Modal App cho GGUF convert (CPU-only, khong GPU)
# ---------------------------------------------------------------------------

gguf_convert_app = modal.App(
    "ai-workers-gguf-converter",
    secrets=[modal.Secret.from_name("hf-token")],
)


# ---------------------------------------------------------------------------
# GGUF model registry — reuse ONNX targets (them GGUF vao cung repo)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GgufModelConfig:
    """Config cho mot model can GGUF convert."""

    name: str  # Registry key
    hf_source: str  # HuggingFace source model ID
    hf_target: str  # HuggingFace target repo (cung repo voi ONNX)
    gguf_name: str  # GGUF filename (e.g. "qwen3-embedding-0.6b")


GGUF_MODELS: dict[str, GgufModelConfig] = {}


def _register(config: GgufModelConfig) -> GgufModelConfig:
    GGUF_MODELS[config.name] = config
    return config


_register(
    GgufModelConfig(
        name="qwen3-embedding-0.6b-gguf",
        hf_source="Qwen/Qwen3-Embedding-0.6B",
        hf_target="n24q02m/Qwen3-Embedding-0.6B-ONNX",
        gguf_name="qwen3-embedding-0.6b",
    )
)

_register(
    GgufModelConfig(
        name="qwen3-reranker-0.6b-gguf",
        hf_source="Qwen/Qwen3-Reranker-0.6B",
        hf_target="n24q02m/Qwen3-Reranker-0.6B-ONNX",
        gguf_name="qwen3-reranker-0.6b",
    )
)


@gguf_convert_app.function(
    image=gguf_converter_image(),
    memory=32768,  # 32GB RAM
    cpu=4.0,
    timeout=3600,  # 1 gio
)
def gguf_convert_model(
    model_name: str,
    hf_source: str,
    hf_target: str,
    gguf_name: str,
    *,
    quant_type: str = "Q4_K_M",
    force: bool = False,
) -> dict[str, object]:
    """Convert mot HuggingFace model sang GGUF va push len HF Hub.

    Chay tren Modal CPU container. Pipeline:
    HF model -> convert_hf_to_gguf.py (F16) -> llama-quantize (Q4_K_M)
    -> push len HuggingFace Hub repo (existing).

    Args:
        model_name: Registry name (e.g. "qwen3-embedding-0.6b-gguf").
        hf_source: Source HuggingFace model ID (e.g. "Qwen/Qwen3-Embedding-0.6B").
        hf_target: Target HuggingFace repo ID (e.g. "n24q02m/Qwen3-Embedding-0.6B-ONNX").
        gguf_name: Base name for GGUF file (e.g. "qwen3-embedding-0.6b").
        quant_type: GGUF quantization type (default "Q4_K_M").
        force: Ghi de neu file da ton tai tren HF Hub.

    Returns:
        Dict chua ket qua: model_name, status, hf_target, gguf_file, size_mb.
    """
    import gc
    import os
    import subprocess
    import tempfile
    from pathlib import Path

    from huggingface_hub import HfApi, list_repo_tree
    from loguru import logger

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        msg = "HF_TOKEN khong duoc set. Can Modal Secret 'hf-token' voi key HF_TOKEN."
        raise ValueError(msg)

    api = HfApi(token=hf_token)

    gguf_filename = f"{gguf_name}-{quant_type.lower().replace('_', '-')}.gguf"
    gguf_repo_path = f"gguf/{gguf_filename}"

    # ------------------------------------------------------------------
    # Kiem tra file da ton tai chua
    # ------------------------------------------------------------------
    if not force:
        try:
            existing_files = [
                f.path for f in list_repo_tree(hf_target, token=hf_token, recursive=True)
            ]
            if gguf_repo_path in existing_files:
                logger.info(
                    "File {} da ton tai trong {}. Bo qua. Dung force=True de ghi de.",
                    gguf_repo_path,
                    hf_target,
                )
                return {
                    "model_name": model_name,
                    "status": "skipped",
                    "reason": "already_exists",
                    "hf_target": hf_target,
                }
        except Exception:
            pass  # Repo chua ton tai, se tao moi

    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir) / "model"
        output_dir = Path(tmpdir) / "output" / "gguf"
        model_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Download model tu HuggingFace Hub
        # ------------------------------------------------------------------
        logger.info("Dang tai model {} tu HuggingFace Hub...", hf_source)

        # Download tat ca files can thiet
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=hf_source,
            local_dir=str(model_dir),
            token=hf_token,
        )
        logger.info("Model downloaded to {}", model_dir)

        # ------------------------------------------------------------------
        # Step 1: convert_hf_to_gguf.py -> F16 GGUF
        # ------------------------------------------------------------------
        f16_path = Path(tmpdir) / f"{gguf_name}-f16.gguf"
        q4_path = output_dir / gguf_filename

        logger.info("Converting HF -> GGUF F16: {}", f16_path)

        convert_cmd = [
            "python",
            "/opt/llama.cpp/convert_hf_to_gguf.py",
            str(model_dir),
            "--outfile",
            str(f16_path),
            "--outtype",
            "f16",
        ]
        result = subprocess.run(
            convert_cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            logger.error("convert_hf_to_gguf.py failed:\n{}", result.stderr)
            msg = f"convert_hf_to_gguf.py failed with exit code {result.returncode}"
            raise RuntimeError(msg)

        f16_size = f16_path.stat().st_size / (1024**2)
        logger.info("GGUF F16 exported: {:.2f} MB", f16_size)

        # Giai phong RAM
        gc.collect()

        # ------------------------------------------------------------------
        # Step 2: llama-quantize -> Q4_K_M
        # ------------------------------------------------------------------
        logger.info("Quantizing GGUF F16 -> {}: {}", quant_type, q4_path)

        quantize_cmd = [
            "/opt/llama.cpp/build/bin/llama-quantize",
            str(f16_path),
            str(q4_path),
            quant_type,
        ]
        result = subprocess.run(
            quantize_cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            logger.error("llama-quantize failed:\n{}", result.stderr)
            msg = f"llama-quantize failed with exit code {result.returncode}"
            raise RuntimeError(msg)

        q4_size = q4_path.stat().st_size / (1024**2)
        logger.info(
            "GGUF {} quantized: {:.2f} MB (compression ratio: {:.1f}x)",
            quant_type,
            q4_size,
            f16_size / q4_size if q4_size > 0 else 0,
        )

        # Xoa F16 intermediate
        f16_path.unlink()

        # ------------------------------------------------------------------
        # Upload GGUF file len HuggingFace Hub
        # ------------------------------------------------------------------
        logger.info("Uploading {} len {}...", gguf_repo_path, hf_target)

        api.create_repo(
            repo_id=hf_target,
            repo_type="model",
            exist_ok=True,
            private=False,
        )

        api.upload_file(
            path_or_fileobj=str(q4_path),
            path_in_repo=gguf_repo_path,
            repo_id=hf_target,
            repo_type="model",
            commit_message=f"Add GGUF {quant_type} converted from {hf_source}",
        )

        logger.info("Push thanh cong len https://huggingface.co/{}", hf_target)

    gc.collect()

    return {
        "model_name": model_name,
        "status": "success",
        "hf_target": hf_target,
        "gguf_file": gguf_repo_path,
        "quant_type": quant_type,
        "size_mb": round(q4_size, 2),
        "url": f"https://huggingface.co/{hf_target}",
    }
