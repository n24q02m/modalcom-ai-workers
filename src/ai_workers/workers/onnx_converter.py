"""Modal CPU function to convert HuggingFace models to ONNX multi-variant.

Runs on a Modal CPU container (32GB RAM) instead of local machine.
Download from HuggingFace Hub -> ONNX export (FP32)
-> INT8 dynamic quantization + INT4 (Q4F16) quantization -> push to HF Hub.

Output structure (fastembed-compatible):
  {hf_repo_id}/
    onnx/model_quantized.onnx    # INT8 dynamic quantized ONNX model
    onnx/model_q4f16.onnx        # INT4 weights + FP16 activations
    config.json                  # Model config (hidden_size, vocab_size, ...)
    tokenizer.json               # Fast tokenizer
    tokenizer_config.json
    special_tokens_map.json
    README.md                    # Model card

Flow:
  CLI (local) -> onnx_convert_model.remote() -> Modal CPU container:
    1. Download model + tokenizer from HuggingFace Hub
    2. Wrap model (only extract the necessary output — last_hidden_state or logits)
    3. torch.onnx.export to FP32 (opset 21, CPU, /tmp/)
    4. onnxruntime.quantization.quantize_dynamic -> INT8 (model_quantized.onnx)
    5. MatMulNBitsQuantizer + float16 -> Q4F16 (model_q4f16.onnx)
    6. Save tokenizer + config + model card to /tmp/
    7. Push all files to HuggingFace Hub (public repo)
"""

from dataclasses import dataclass

import modal

from ai_workers.common.images import onnx_converter_image

# ---------------------------------------------------------------------------
# Modal App for ONNX conversion (CPU-only, no GPU)
# ---------------------------------------------------------------------------

onnx_convert_app = modal.App(
    "ai-workers-onnx-converter",
    secrets=[modal.Secret.from_name("hf-token")],
)


# ---------------------------------------------------------------------------
# ONNX model registry — models available for conversion in the qwen3-embed package
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OnnxModelConfig:
    """Configuration for a model that needs ONNX conversion."""

    name: str  # Registry key
    hf_source: str  # HuggingFace source model ID
    hf_target: str  # HuggingFace target repo for ONNX output
    model_class: str  # "AutoModel" | "AutoModelForCausalLM"
    output_attr: str  # "last_hidden_state" | "logits"
    trust_remote_code: bool = False


ONNX_MODELS: dict[str, OnnxModelConfig] = {}


def _register(config: OnnxModelConfig) -> OnnxModelConfig:
    ONNX_MODELS[config.name] = config
    return config


_register(
    OnnxModelConfig(
        name="qwen3-embedding-0.6b-onnx",
        hf_source="Qwen/Qwen3-Embedding-0.6B",
        hf_target="n24q02m/Qwen3-Embedding-0.6B-ONNX",
        model_class="AutoModel",
        output_attr="last_hidden_state",
    )
)

_register(
    OnnxModelConfig(
        name="qwen3-reranker-0.6b-onnx",
        hf_source="Qwen/Qwen3-Reranker-0.6B",
        hf_target="n24q02m/Qwen3-Reranker-0.6B-ONNX",
        model_class="AutoModelForCausalLM",
        output_attr="logits",
    )
)

_register(
    OnnxModelConfig(
        name="qwen3-reranker-0.6b-onnx-yesno",
        hf_source="Qwen/Qwen3-Reranker-0.6B",
        hf_target="n24q02m/Qwen3-Reranker-0.6B-ONNX-YesNo",
        model_class="AutoModelForCausalLM",
        output_attr="yesno_logits",
    )
)


# ---------------------------------------------------------------------------
# Model card template
# ---------------------------------------------------------------------------

_MODEL_CARD_TEMPLATE = """\
---
license: apache-2.0
tags:
  - onnx
  - quantized
  - qwen3
  - {model_type}
base_model: {hf_source}
pipeline_tag: {pipeline_tag}
---

# {hf_target}

ONNX-optimized version of [{hf_source}](https://huggingface.co/{hf_source})
for use with [qwen3-embed](https://github.com/n24q02m/qwen3-embed)
and [fastembed](https://github.com/qdrant/fastembed).

## Available Variants

| Variant | File | Size | Description |
|---------|------|------|-------------|
| INT8 | `onnx/model_quantized.onnx` | {int8_size_mb:.0f} MB | Dynamic INT8 quantization (default) |
| Q4F16 | `onnx/model_q4f16.onnx` | {q4f16_size_mb:.0f} MB | INT4 weights + FP16 activations |

## Usage

```python
# INT8 (default)
from qwen3_embed import {usage_class}
model = {usage_class}("Qwen/{model_short}")

# Q4F16 (smaller, slightly less accurate)
model = {usage_class}("Qwen/{model_short}-Q4F16")
```

## Conversion Details

- **Source**: {hf_source}
- **ONNX opset**: 21
- **INT8**: `onnxruntime.quantization.quantize_dynamic` (QInt8)
- **Q4F16**: `MatMulNBitsQuantizer` (block_size=128, symmetric) + FP16 cast

## Related

- GGUF variants: [{gguf_repo}](https://huggingface.co/{gguf_repo})
"""


def _generate_model_card(
    config: OnnxModelConfig,
    int8_size_mb: float,
    q4f16_size_mb: float,
) -> str:
    """Generate a model card README.md for the HuggingFace repo."""
    is_embedding = config.output_attr == "last_hidden_state"
    model_short = config.hf_source.split("/")[-1]
    gguf_repo = config.hf_target.replace("-ONNX", "-GGUF")

    return _MODEL_CARD_TEMPLATE.format(
        hf_source=config.hf_source,
        hf_target=config.hf_target,
        model_type="text-embedding" if is_embedding else "text-reranking",
        pipeline_tag="feature-extraction" if is_embedding else "text-classification",
        int8_size_mb=int8_size_mb,
        q4f16_size_mb=q4f16_size_mb,
        usage_class="TextEmbedding" if is_embedding else "TextCrossEncoder",
        model_short=model_short,
        gguf_repo=gguf_repo,
    )


@onnx_convert_app.function(
    image=onnx_converter_image(),
    memory=32768,  # 32GB RAM
    cpu=4.0,
    timeout=3600,  # 1 hour
)
def onnx_convert_model(
    model_name: str,
    hf_source: str,
    hf_target: str,
    model_class: str,
    output_attr: str,
    *,
    trust_remote_code: bool = False,
    opset_version: int = 21,
    force: bool = False,
) -> dict[str, object]:
    """Convert a HuggingFace model to ONNX multi-variant and push to HF Hub.

    Runs on a Modal CPU container. Pipeline:
    ONNX FP32 export -> INT8 + Q4F16 quantization
    -> push public repo to HuggingFace Hub.

    Args:
        model_name: Registry name (e.g. "qwen3-embedding-0.6b-onnx").
        hf_source: Source HuggingFace model ID (e.g. "Qwen/Qwen3-Embedding-0.6B").
        hf_target: Target HuggingFace repo ID (e.g. "n24q02m/Qwen3-Embedding-0.6B-ONNX").
        model_class: "AutoModel" or "AutoModelForCausalLM".
        output_attr: Attribute to extract from model output ("last_hidden_state" | "logits").
        opset_version: ONNX opset version (default 21, required for MatMulNBits).
        force: Overwrite if repo already exists on HF Hub.

    Returns:
        Dict containing results: model_name, status, hf_target, variants, total_size_mb.
    """
    import gc
    import os
    import tempfile
    from pathlib import Path

    import onnx
    import torch
    from huggingface_hub import HfApi, repo_exists
    from loguru import logger
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        msg = "HF_TOKEN is not set. Requires Modal Secret 'hf-token' with key HF_TOKEN."
        raise ValueError(msg)

    api = HfApi(token=hf_token)

    # ------------------------------------------------------------------
    # Check if repo already exists
    # ------------------------------------------------------------------
    if not force and repo_exists(hf_target, token=hf_token):
        logger.info(
            "Repo {} already exists on HF Hub. Skipping. Use force=True to overwrite.",
            hf_target,
        )
        return {
            "model_name": model_name,
            "status": "skipped",
            "reason": "already_exists",
            "hf_target": hf_target,
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "output"
        onnx_dir = output_path / "onnx"
        output_path.mkdir(parents=True, exist_ok=True)
        onnx_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Download model + tokenizer from HuggingFace Hub
        # ------------------------------------------------------------------
        logger.info("Loading model {} from HuggingFace Hub...", hf_source)

        tokenizer = AutoTokenizer.from_pretrained(hf_source, trust_remote_code=trust_remote_code)

        model_cls_map: dict[str, type] = {
            "AutoModel": AutoModel,
            "AutoModelForCausalLM": AutoModelForCausalLM,
        }
        cls = model_cls_map.get(model_class)
        if cls is None:
            msg = (
                f"Model class '{model_class}' is invalid. Choose from: {list(model_cls_map.keys())}"
            )
            raise ValueError(msg)

        model = cls.from_pretrained(
            hf_source,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.float32,  # FP32 for accurate quantization
            low_cpu_mem_usage=True,
            device_map="cpu",
        )
        model.config.use_cache = False  # Disable KV cache
        model.eval()
        logger.info("Model loaded: {} params", sum(p.numel() for p in model.parameters()))

        # ------------------------------------------------------------------
        # Wrap model — only output the necessary tensor
        # ------------------------------------------------------------------
        class _OnnxWrapper(torch.nn.Module):
            """Wrapper that retains exactly 1 output tensor for ONNX export."""

            def __init__(self, inner: torch.nn.Module, attr: str) -> None:
                super().__init__()
                self.inner = inner
                self.attr = attr

            def forward(
                self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
            ) -> torch.Tensor:
                out = self.inner(input_ids=input_ids, attention_mask=attention_mask)
                return getattr(out, self.attr)

        class _YesNoWrapper(torch.nn.Module):
            """Wrapper that outputs only [no, yes] logits at the last token.

            Reduces output from (batch, seq_len, vocab_size) to (batch, 2),
            cutting runtime memory by ~150x for Qwen3 reranker models.
            """

            TOKEN_YES_ID = 9693
            TOKEN_NO_ID = 2152

            def __init__(self, inner: torch.nn.Module) -> None:
                super().__init__()
                self.model = inner.model  # Transformer backbone
                lm_head_weight = inner.lm_head.weight.data  # (vocab, hidden)
                self.yes_no_head = torch.nn.Linear(lm_head_weight.shape[1], 2, bias=False)
                self.yes_no_head.weight.data = lm_head_weight[
                    [self.TOKEN_NO_ID, self.TOKEN_YES_ID], :
                ]

            def forward(
                self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
            ) -> torch.Tensor:
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden = out.last_hidden_state[:, -1, :]  # (batch, hidden)
                return self.yes_no_head(last_hidden)  # (batch, 2)

        if output_attr == "yesno_logits":
            wrapper = _YesNoWrapper(model)
            # Override output_attr for ONNX naming
            onnx_output_name = "logits"
        else:
            wrapper = _OnnxWrapper(model, output_attr)
            onnx_output_name = output_attr

        # ------------------------------------------------------------------
        # Create dummy input for tracing
        # ------------------------------------------------------------------
        dummy = tokenizer("hello world", return_tensors="pt")
        dummy_ids = dummy["input_ids"]
        dummy_mask = dummy["attention_mask"]

        # ------------------------------------------------------------------
        # Export ONNX FP32 (opset 21 for MatMulNBits support)
        # ------------------------------------------------------------------
        fp32_path = Path(tmpdir) / "model_fp32.onnx"
        int8_path = onnx_dir / "model_quantized.onnx"
        q4f16_path = onnx_dir / "model_q4f16.onnx"

        logger.info("Exporting ONNX FP32 -> {} (opset {})", fp32_path, opset_version)

        # YesNo output is (batch, 2) — no sequence_length axis
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        }
        if output_attr == "yesno_logits":
            dynamic_axes[onnx_output_name] = {0: "batch_size"}
        else:
            dynamic_axes[onnx_output_name] = {0: "batch_size", 1: "sequence_length"}

        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (dummy_ids, dummy_mask),
                str(fp32_path),
                input_names=["input_ids", "attention_mask"],
                output_names=[onnx_output_name],
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True,
            )

        fp32_size = fp32_path.stat().st_size / (1024**2)
        logger.info("ONNX FP32 exported: {:.2f} MB", fp32_size)

        # Check for external data files (torch.onnx.export creates them for large models)
        fp32_data_path = fp32_path.with_suffix(".onnx.data")
        if fp32_data_path.exists():
            fp32_total_size = (fp32_path.stat().st_size + fp32_data_path.stat().st_size) / (1024**2)
            logger.info(
                "  Graph: {:.2f} MB + External data: {:.2f} MB = Total: {:.2f} MB",
                fp32_size,
                fp32_data_path.stat().st_size / (1024**2),
                fp32_total_size,
            )
        else:
            fp32_total_size = fp32_size

        # Free model (need RAM for quantization)
        del model, wrapper
        gc.collect()

        # ------------------------------------------------------------------
        # Variant 1: INT8 dynamic quantization -> model_quantized.onnx
        # ------------------------------------------------------------------
        logger.info("Quantizing ONNX FP32 -> INT8: {}", int8_path)
        quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(int8_path),
            weight_type=QuantType.QInt8,
        )

        int8_size = int8_path.stat().st_size / (1024**2)
        logger.info(
            "INT8 quantized: {:.2f} MB (compression ratio: {:.1f}x)",
            int8_size,
            fp32_total_size / int8_size if int8_size > 0 else 0,
        )

        # ------------------------------------------------------------------
        # Variant 2: Q4F16 (INT4 weights + FP16 activations) -> model_q4f16.onnx
        # ------------------------------------------------------------------
        logger.info("Quantizing ONNX FP32 -> Q4F16: {}", q4f16_path)

        # Step 2a: INT4 weight quantization via MatMulNBitsQuantizer
        quantizer = MatMulNBitsQuantizer(
            model=str(fp32_path),
            bits=4,
            block_size=128,
            is_symmetric=True,
            accuracy_level=4,
        )
        quantizer.process()

        # Step 2b: Cast remaining FP32 tensors to FP16
        from onnxconverter_common import float16

        q4f16_model = float16.convert_float_to_float16(
            quantizer.model.model,
            keep_io_types=False,  # Full float16 (including I/O)
        )

        # Fix: convert_float_to_float16 doesn't update Cast op's 'to' attribute.
        # Internal Cast(to=float32) nodes create type mismatches with float16 tensors.
        # Force all Cast(to=float32) -> Cast(to=float16) for type consistency.
        # TensorProto.FLOAT = 1, TensorProto.FLOAT16 = 10
        for node in q4f16_model.graph.node:
            if node.op_type == "Cast":
                for attr in node.attribute:
                    if attr.name == "to" and attr.i == onnx.TensorProto.FLOAT:
                        attr.i = onnx.TensorProto.FLOAT16

        # Clear stale type annotations — OnnxRuntime infers types from ops at load.
        q4f16_model.graph.ClearField("value_info")

        onnx.save(q4f16_model, str(q4f16_path))

        del quantizer, q4f16_model
        gc.collect()

        q4f16_size = q4f16_path.stat().st_size / (1024**2)
        logger.info(
            "Q4F16 quantized: {:.2f} MB (compression ratio: {:.1f}x)",
            q4f16_size,
            fp32_total_size / q4f16_size if q4f16_size > 0 else 0,
        )

        # Delete FP32 model + external data (not uploaded)
        fp32_path.unlink()
        if fp32_data_path.exists():
            fp32_data_path.unlink()

        # ------------------------------------------------------------------
        # Save tokenizer + config
        # ------------------------------------------------------------------
        logger.info("Saving tokenizer + config -> {}", output_path)
        tokenizer.save_pretrained(str(output_path))

        config = AutoConfig.from_pretrained(hf_source, trust_remote_code=trust_remote_code)
        config.save_pretrained(str(output_path))

        # ------------------------------------------------------------------
        # Generate model card README.md
        # ------------------------------------------------------------------
        model_card = _generate_model_card(
            OnnxModelConfig(
                name=model_name,
                hf_source=hf_source,
                hf_target=hf_target,
                model_class=model_class,
                output_attr=output_attr,
                trust_remote_code=trust_remote_code,
            ),
            int8_size_mb=int8_size,
            q4f16_size_mb=q4f16_size,
        )
        readme_path = output_path / "README.md"
        readme_path.write_text(model_card, encoding="utf-8")
        logger.info("Model card generated: {}", readme_path)

        # ------------------------------------------------------------------
        # File statistics
        # ------------------------------------------------------------------
        total_size = 0.0
        file_count = 0
        for f in sorted(output_path.rglob("*")):
            if f.is_file():
                size_mb = f.stat().st_size / (1024**2)
                total_size += size_mb
                file_count += 1
                logger.info("  {} ({:.2f} MB)", f.relative_to(output_path), size_mb)

        logger.info(
            "Conversion complete: {} files ({:.2f} MB), pushing to {}...",
            file_count,
            total_size,
            hf_target,
        )

        # ------------------------------------------------------------------
        # Create repo + push to HuggingFace Hub
        # ------------------------------------------------------------------
        api.create_repo(
            repo_id=hf_target,
            repo_type="model",
            exist_ok=True,
            private=False,  # Public repo
        )

        api.upload_folder(
            repo_id=hf_target,
            folder_path=str(output_path),
            repo_type="model",
            commit_message=f"Add ONNX INT8 + Q4F16 models converted from {hf_source}",
            delete_patterns=["onnx/*.onnx"],  # Delete old ONNX files (model.onnx) before upload
        )

        logger.info("Successfully pushed to https://huggingface.co/{}", hf_target)

    gc.collect()

    return {
        "model_name": model_name,
        "status": "success",
        "hf_target": hf_target,
        "variants": {
            "int8": {"file": "onnx/model_quantized.onnx", "size_mb": round(int8_size, 2)},
            "q4f16": {"file": "onnx/model_q4f16.onnx", "size_mb": round(q4f16_size, 2)},
        },
        "files_count": file_count,
        "total_size_mb": round(total_size, 2),
        "url": f"https://huggingface.co/{hf_target}",
    }
