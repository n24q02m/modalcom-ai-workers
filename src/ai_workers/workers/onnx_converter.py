"""Modal CPU function để convert HuggingFace models sang ONNX INT8.

Chạy trên Modal CPU container (32GB RAM) thay vì máy local.
Download từ HuggingFace Hub → ONNX export (FP32) → INT8 quantization → push HF Hub.

Output structure (fastembed-compatible):
  {hf_repo_id}/
    onnx/model.onnx              # INT8 quantized ONNX model
    config.json                  # Model config (hidden_size, vocab_size, ...)
    tokenizer.json               # Fast tokenizer
    tokenizer_config.json
    special_tokens_map.json

Flow:
  CLI (local) → onnx_convert_model.remote() → Modal CPU container:
    1. Download model + tokenizer từ HuggingFace Hub
    2. Wrap model (chỉ lấy output cần thiết — last_hidden_state hoặc logits)
    3. torch.onnx.export sang FP32 (CPU, /tmp/)
    4. onnxruntime.quantization.quantize_dynamic → INT8 (/tmp/)
    5. Lưu tokenizer + config vào /tmp/
    6. Push toàn bộ lên HuggingFace Hub (public repo)
"""

from __future__ import annotations

from dataclasses import dataclass

import modal

from ai_workers.common.images import onnx_converter_image

# ---------------------------------------------------------------------------
# Modal App cho ONNX convert (CPU-only, không GPU)
# ---------------------------------------------------------------------------

onnx_convert_app = modal.App(
    "ai-workers-onnx-converter",
    secrets=[modal.Secret.from_name("hf-token")],
)


# ---------------------------------------------------------------------------
# ONNX model registry — models cần convert cho qwen3-embed package
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OnnxModelConfig:
    """Config cho một model cần ONNX convert."""

    name: str  # Registry key
    hf_source: str  # HuggingFace source model ID
    hf_target: str  # HuggingFace target repo cho ONNX output
    model_class: str  # "AutoModel" | "AutoModelForCausalLM"
    output_attr: str  # "last_hidden_state" | "logits"


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


@onnx_convert_app.function(
    image=onnx_converter_image(),
    memory=32768,  # 32GB RAM
    cpu=4.0,
    timeout=3600,  # 1 giờ
)
def onnx_convert_model(
    model_name: str,
    hf_source: str,
    hf_target: str,
    model_class: str,
    output_attr: str,
    *,
    opset_version: int = 17,
    force: bool = False,
) -> dict[str, object]:
    """Convert một HuggingFace model sang ONNX INT8 và push lên HF Hub.

    Chạy trên Modal CPU container. ONNX FP32 export → INT8 dynamic
    quantization → push public repo lên HuggingFace Hub.

    Args:
        model_name: Registry name (e.g. "qwen3-embedding-0.6b-onnx").
        hf_source: Source HuggingFace model ID (e.g. "Qwen/Qwen3-Embedding-0.6B").
        hf_target: Target HuggingFace repo ID (e.g. "n24q02m/Qwen3-Embedding-0.6B-ONNX").
        model_class: "AutoModel" hoặc "AutoModelForCausalLM".
        output_attr: Attribute lấy từ model output ("last_hidden_state" | "logits").
        opset_version: ONNX opset version (default 17).
        force: Ghi đè nếu repo đã tồn tại trên HF Hub.

    Returns:
        Dict chứa kết quả: model_name, status, hf_target, files_count, total_size_mb.
    """
    import gc
    import os
    import tempfile
    from pathlib import Path

    import torch
    from huggingface_hub import HfApi, repo_exists
    from loguru import logger
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        msg = "HF_TOKEN không được set. Cần Modal Secret 'hf-token' với key HF_TOKEN."
        raise ValueError(msg)

    api = HfApi(token=hf_token)

    # ------------------------------------------------------------------
    # Kiểm tra repo đã tồn tại chưa
    # ------------------------------------------------------------------
    if not force and repo_exists(hf_target, token=hf_token):
        logger.info(
            "Repo {} đã tồn tại trên HF Hub. Bỏ qua. Dùng force=True để ghi đè.",
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
        # Download model + tokenizer từ HuggingFace Hub
        # ------------------------------------------------------------------
        logger.info("Đang tải model {} từ HuggingFace Hub...", hf_source)

        tokenizer = AutoTokenizer.from_pretrained(hf_source, trust_remote_code=True)

        model_cls_map: dict[str, type] = {
            "AutoModel": AutoModel,
            "AutoModelForCausalLM": AutoModelForCausalLM,
        }
        cls = model_cls_map.get(model_class)
        if cls is None:
            msg = f"Model class '{model_class}' không hợp lệ. Chọn: {list(model_cls_map.keys())}"
            raise ValueError(msg)

        model = cls.from_pretrained(
            hf_source,
            trust_remote_code=True,
            torch_dtype=torch.float32,  # FP32 cho quantization chính xác
            low_cpu_mem_usage=True,
            device_map="cpu",
        )
        model.config.use_cache = False  # Tắt KV cache
        model.eval()
        logger.info("Model loaded: {} params", sum(p.numel() for p in model.parameters()))

        # ------------------------------------------------------------------
        # Wrap model — chỉ output tensor cần thiết
        # ------------------------------------------------------------------
        class _OnnxWrapper(torch.nn.Module):
            """Wrapper giữ lại chính xác 1 output tensor cho ONNX export."""

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

        wrapper = _OnnxWrapper(model, output_attr)

        # ------------------------------------------------------------------
        # Tạo dummy input cho tracing
        # ------------------------------------------------------------------
        dummy = tokenizer("hello world", return_tensors="pt")
        dummy_ids = dummy["input_ids"]
        dummy_mask = dummy["attention_mask"]

        # ------------------------------------------------------------------
        # Export ONNX FP32
        # ------------------------------------------------------------------
        fp32_path = Path(tmpdir) / "model_fp32.onnx"
        int8_path = onnx_dir / "model.onnx"

        logger.info("Exporting ONNX FP32 → {} (opset {})", fp32_path, opset_version)

        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (dummy_ids, dummy_mask),
                str(fp32_path),
                input_names=["input_ids", "attention_mask"],
                output_names=[output_attr],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    output_attr: {0: "batch_size", 1: "sequence_length"},
                },
                opset_version=opset_version,
                do_constant_folding=True,
            )

        fp32_size = fp32_path.stat().st_size / (1024**2)
        logger.info("ONNX FP32 exported: {:.2f} MB", fp32_size)

        # Giải phóng model (cần RAM cho quantization)
        del model, wrapper
        gc.collect()

        # ------------------------------------------------------------------
        # INT8 dynamic quantization
        # ------------------------------------------------------------------
        logger.info("Quantizing ONNX FP32 → INT8: {}", int8_path)
        quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(int8_path),
            weight_type=QuantType.QInt8,
        )

        int8_size = int8_path.stat().st_size / (1024**2)
        logger.info(
            "INT8 quantized: {:.2f} MB (compression ratio: {:.1f}x)",
            int8_size,
            fp32_size / int8_size if int8_size > 0 else 0,
        )

        # Xoá FP32 model
        fp32_path.unlink()

        # ------------------------------------------------------------------
        # Lưu tokenizer + config
        # ------------------------------------------------------------------
        logger.info("Saving tokenizer + config → {}", output_path)
        tokenizer.save_pretrained(str(output_path))

        config = AutoConfig.from_pretrained(hf_source, trust_remote_code=True)
        config.save_pretrained(str(output_path))

        # ------------------------------------------------------------------
        # Thống kê files
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
            "Convert xong {} files ({:.2f} MB), đang push lên {}...",
            file_count,
            total_size,
            hf_target,
        )

        # ------------------------------------------------------------------
        # Tạo repo + push lên HuggingFace Hub
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
            commit_message=f"Add ONNX INT8 model converted from {hf_source}",
        )

        logger.info("Push thành công lên https://huggingface.co/{}", hf_target)

    gc.collect()

    return {
        "model_name": model_name,
        "status": "success",
        "hf_target": hf_target,
        "files_count": file_count,
        "total_size_mb": round(total_size, 2),
        "url": f"https://huggingface.co/{hf_target}",
    }
