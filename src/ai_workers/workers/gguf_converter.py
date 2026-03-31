"""Modal CPU function to convert HuggingFace models to GGUF Q4_K_M.

Runs on a Modal CPU container (32GB RAM) instead of local machine.
Download from HuggingFace Hub -> convert_hf_to_gguf.py (F16) -> llama-quantize (Q4_K_M)
-> push to HF Hub.

Output structure:
  {hf_repo_id}/
    gguf/{model_name}-q4_k_m.gguf   # GGUF Q4_K_M quantized

Flow:
  CLI (local) -> gguf_convert_model.remote() -> Modal CPU container:
    1. Download model from HuggingFace Hub (safetensors)
    2. Run convert_hf_to_gguf.py -> F16 GGUF (intermediate)
    3. Run llama-quantize -> Q4_K_M GGUF
    4. Push to HuggingFace Hub (existing repo)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from huggingface_hub import HfApi

from dataclasses import dataclass

import modal

from ai_workers.common.images import gguf_converter_image

# ---------------------------------------------------------------------------
# Modal App for GGUF conversion (CPU-only, no GPU)
# ---------------------------------------------------------------------------

gguf_convert_app = modal.App(
    "ai-workers-gguf-converter",
    secrets=[modal.Secret.from_name("hf-token")],
)


# ---------------------------------------------------------------------------
# GGUF model registry — dedicated *-GGUF repos (separate from ONNX)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GgufModelConfig:
    """Configuration for a model that needs GGUF conversion."""

    name: str  # Registry key
    hf_source: str  # HuggingFace source model ID
    hf_target: str  # HuggingFace target repo (dedicated *-GGUF repo)
    gguf_name: str  # GGUF filename (e.g. "qwen3-embedding-0.6b")
    output_attr: str  # "last_hidden_state" (embedding) or "logits" (reranker)


GGUF_MODELS: dict[str, GgufModelConfig] = {}


def _register(config: GgufModelConfig) -> GgufModelConfig:
    GGUF_MODELS[config.name] = config
    return config


_register(
    GgufModelConfig(
        name="qwen3-embedding-0.6b-gguf",
        hf_source="Qwen/Qwen3-Embedding-0.6B",
        hf_target="n24q02m/Qwen3-Embedding-0.6B-GGUF",
        gguf_name="qwen3-embedding-0.6b",
        output_attr="last_hidden_state",
    )
)

_register(
    GgufModelConfig(
        name="qwen3-reranker-0.6b-gguf",
        hf_source="Qwen/Qwen3-Reranker-0.6B",
        hf_target="n24q02m/Qwen3-Reranker-0.6B-GGUF",
        gguf_name="qwen3-reranker-0.6b",
        output_attr="logits",
    )
)


# ---------------------------------------------------------------------------
# Model card template
# ---------------------------------------------------------------------------

_GGUF_MODEL_CARD_TEMPLATE = """\
---
license: apache-2.0
tags:
  - gguf
  - quantized
  - qwen3
  - {model_type}
base_model: {hf_source}
pipeline_tag: {pipeline_tag}
---

# {hf_target}

GGUF-quantized version of [{hf_source}](https://huggingface.co/{hf_source})
for use with [qwen3-embed](https://github.com/n24q02m/qwen3-embed)
and [llama-cpp-python](https://github.com/abetlen/llama-cpp-python).

## Available Variants

| Variant | File | Size | Description |
|---------|------|------|-------------|
| Q4_K_M | `{gguf_filename}` | {size_mb:.0f} MB | 4-bit quantization (recommended) |

## Usage

```python
from qwen3_embed import {usage_class}

model = {usage_class}("Qwen/{model_short}-GGUF")
```

Requires optional dependency:

```bash
pip install qwen3-embed[gguf]
```

## Conversion Details

- **Source**: {hf_source}
- **Method**: `convert_hf_to_gguf.py` (F16) + `llama-quantize` (Q4_K_M)

## Related

- ONNX variants: [{onnx_repo}](https://huggingface.co/{onnx_repo})
"""


def _generate_gguf_model_card(
    config: GgufModelConfig,
    gguf_filename: str,
    size_mb: float,
) -> str:
    """Generate a model card README.md for the GGUF HuggingFace repo."""
    is_embedding = config.output_attr == "last_hidden_state"
    model_short = config.hf_source.split("/")[-1]
    onnx_repo = config.hf_target.replace("-GGUF", "-ONNX")

    return _GGUF_MODEL_CARD_TEMPLATE.format(
        hf_source=config.hf_source,
        hf_target=config.hf_target,
        model_type="text-embedding" if is_embedding else "text-reranking",
        pipeline_tag="feature-extraction" if is_embedding else "text-classification",
        gguf_filename=gguf_filename,
        size_mb=size_mb,
        usage_class="TextEmbedding" if is_embedding else "TextCrossEncoder",
        model_short=model_short,
        onnx_repo=onnx_repo,
    )



def _check_if_gguf_exists(
    hf_target: str, gguf_repo_path: str, hf_token: str, force: bool
) -> bool:
    """Check if the GGUF file already exists on the HuggingFace Hub."""
    from huggingface_hub import list_repo_tree
    from loguru import logger

    if force:
        return False

    try:
        existing_files = [
            f.path for f in list_repo_tree(hf_target, token=hf_token, recursive=True)
        ]
        if gguf_repo_path in existing_files:
            logger.info(
                "File {} already exists in {}. Skipping. Use force=True to overwrite.",
                gguf_repo_path,
                hf_target,
            )
            return True
    except Exception:
        pass  # Repo does not exist yet, will be created

    return False

def _convert_hf_to_f16(model_dir: Path, f16_path: Path) -> float:
    """Run convert_hf_to_gguf.py to create intermediate F16 GGUF file."""
    import subprocess

    from loguru import logger

    logger.info("Converting HF -> GGUF F16: {}", f16_path)

    convert_cmd = [
        "python",
        "/opt/llama.cpp/convert_hf_to_gguf.py",
        str(model_dir.resolve()),
        "--outfile",
        str(f16_path.resolve()),
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
    return f16_size

def _quantize_f16_to_q4(f16_path: Path, q4_path: Path, quant_type: str, f16_size: float) -> float:
    """Run llama-quantize to create the final Q4_K_M GGUF file."""
    import subprocess

    from loguru import logger

    logger.info("Quantizing GGUF F16 -> {}: {}", quant_type, q4_path)

    quantize_cmd = [
        "/opt/llama.cpp/build/bin/llama-quantize",
        str(f16_path.resolve()),
        str(q4_path.resolve()),
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
    return q4_size

def _upload_gguf_to_hf(
    api: HfApi,
    hf_source: str,
    hf_target: str,
    gguf_repo_path: str,
    q4_path: Path,
    quant_type: str,
    model_card: str,
    hf_token: str,
) -> None:
    """Upload the GGUF file, model card, and configs to the HuggingFace Hub."""
    from loguru import logger

    logger.info("Uploading {} to {}...", gguf_repo_path, hf_target)

    api.create_repo(
        repo_id=hf_target,
        repo_type="model",
        exist_ok=True,
        private=False,
    )

    api.upload_file(
        path_or_fileobj=str(q4_path.resolve()),
        path_in_repo=gguf_repo_path,
        repo_id=hf_target,
        repo_type="model",
        commit_message=f"Add GGUF {quant_type} converted from {hf_source}",
    )

    api.upload_file(
        path_or_fileobj=model_card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=hf_target,
        repo_type="model",
        commit_message="docs: update model card",
    )

    # Upload tokenizer + config from source for model loading
    from huggingface_hub import hf_hub_download as _hf_download

    for cfg_file in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
        try:
            local_cfg = _hf_download(repo_id=hf_source, filename=cfg_file, token=hf_token)
            api.upload_file(
                path_or_fileobj=local_cfg,
                path_in_repo=cfg_file,
                repo_id=hf_target,
                repo_type="model",
                commit_message=f"Add {cfg_file}",
            )
        except Exception:
            pass  # Some config files may not exist

    logger.info("Successfully pushed to https://huggingface.co/{}", hf_target)




@gguf_convert_app.function(
    image=gguf_converter_image(),
    memory=32768,  # 32GB RAM
    cpu=4.0,
    timeout=3600,  # 1 hour
)
def gguf_convert_model(
    model_name: str,
    hf_source: str,
    hf_target: str,
    gguf_name: str,
    output_attr: str = "last_hidden_state",
    *,
    quant_type: str = "Q4_K_M",
    force: bool = False,
) -> dict[str, object]:
    """Convert a HuggingFace model to GGUF and push to HF Hub.

    Runs on a Modal CPU container. Pipeline:
    HF model -> convert_hf_to_gguf.py (F16) -> llama-quantize (Q4_K_M)
    -> push to HuggingFace Hub repo (existing).

    Args:
        model_name: Registry name (e.g. "qwen3-embedding-0.6b-gguf").
        hf_source: Source HuggingFace model ID (e.g. "Qwen/Qwen3-Embedding-0.6B").
        hf_target: Target HuggingFace repo ID (e.g. "n24q02m/Qwen3-Embedding-0.6B-GGUF").
        gguf_name: Base name for GGUF file (e.g. "qwen3-embedding-0.6b").
        output_attr: Model output attribute ("last_hidden_state" or "logits").
        quant_type: GGUF quantization type (default "Q4_K_M").
        force: Overwrite if file already exists on HF Hub.

    Returns:
        Dict containing results: model_name, status, hf_target, gguf_file, size_mb.
    """
    import gc
    import os
    import re
    import tempfile
    from pathlib import Path

    from huggingface_hub import HfApi
    from loguru import logger

    if not re.match(r"^[a-zA-Z0-9_.-]+$", gguf_name):
        raise ValueError(f"Invalid gguf_name: {gguf_name}")
    if not re.match(r"^[a-zA-Z0-9_.-]+$", quant_type):
        raise ValueError(f"Invalid quant_type: {quant_type}")
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        msg = "HF_TOKEN is not set. Requires Modal Secret 'hf-token' with key HF_TOKEN."
        raise ValueError(msg)

    api = HfApi(token=hf_token)

    gguf_filename = f"{gguf_name}-{quant_type.lower().replace('_', '-')}.gguf"
    gguf_repo_path = gguf_filename  # Root level in dedicated GGUF repo

    if _check_if_gguf_exists(hf_target, gguf_repo_path, hf_token, force):
        return {
            "model_name": model_name,
            "status": "skipped",
            "reason": "already_exists",
            "hf_target": hf_target,
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir) / "model"
        output_dir = Path(tmpdir) / "output"
        model_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading model {} from HuggingFace Hub...", hf_source)

        # Download all required files
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=hf_source,
            local_dir=str(model_dir.resolve()),
            token=hf_token,
        )
        logger.info("Model downloaded to {}", model_dir)

        f16_path = Path(tmpdir) / f"{gguf_name}-f16.gguf"
        q4_path = Path(tmpdir) / "output" / gguf_filename

        f16_size = _convert_hf_to_f16(model_dir, f16_path)
        gc.collect()

        q4_size = _quantize_f16_to_q4(f16_path, q4_path, quant_type, f16_size)
        f16_path.unlink()

        # Upload model card
        config_obj = GgufModelConfig(
            name=model_name,
            hf_source=hf_source,
            hf_target=hf_target,
            gguf_name=gguf_name,
            output_attr=output_attr,
        )
        model_card = _generate_gguf_model_card(config_obj, gguf_filename, q4_size)

        _upload_gguf_to_hf(
            api,
            hf_source,
            hf_target,
            gguf_repo_path,
            q4_path,
            quant_type,
            model_card,
            hf_token,
        )

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
