"""[DEPRECATED] Modal CPU function de convert HuggingFace models sang SafeTensors.

DEPRECATED: Workers gio tai model truc tiep tu HuggingFace Hub qua Xet protocol.
Khong can convert sang R2 nua. Module nay chi giu lai de backward compatibility.

Neu can convert sang ONNX INT8, dung onnx_converter.py thay the.

Flow cu (DEPRECATED):
  CLI (local) -> converter.convert_model.remote() -> Modal CPU container:
    1. Download model tu HuggingFace Hub (container co internet)
    2. Load model + ep kieu dtype (CPU, 32GB RAM)
    3. Luu SafeTensors vao CloudBucketMount (ghi thang R2)
    4. Tra ve ket qua (so file, tong dung luong)
"""

from __future__ import annotations

import modal

from ai_workers.common.images import MODELS_MOUNT_PATH, converter_image
from ai_workers.common.r2 import get_modal_cloud_bucket_mount

# ---------------------------------------------------------------------------
# Modal App cho convert (CPU-only, không GPU)
# ---------------------------------------------------------------------------

r2_mount_writable = get_modal_cloud_bucket_mount(read_only=False)

convert_app = modal.App(
    "ai-workers-converter",
    secrets=[modal.Secret.from_name("r2-credentials")],
)


@convert_app.function(
    image=converter_image(),
    volumes={MODELS_MOUNT_PATH: r2_mount_writable},
    memory=32768,  # 32GB RAM (đủ cho model 8B FP16)
    cpu=4.0,
    timeout=3600,  # 1 giờ
)
def convert_model(
    model_name: str,
    hf_id: str,
    precision: str,
    model_class: str,
    task: str,
    trust_remote_code: bool,
    extra_load_kwargs: dict[str, object],
    *,
    force: bool = False,
) -> dict[str, object]:
    """Convert một model HuggingFace sang SafeTensors và ghi vào R2.

    Chạy trên Modal CPU container. Kết quả được ghi trực tiếp vào
    R2 bucket qua CloudBucketMount (writable).

    Args:
        model_name: Tên model trong registry (dùng làm R2 prefix).
        hf_id: HuggingFace model ID (e.g. "Qwen/Qwen3-Embedding-0.6B").
        precision: "fp16" hoặc "bf16".
        model_class: "AutoModel", "AutoModelForCausalLM", hoặc "AutoModelForSpeechSeq2Seq".
        task: Task string (e.g. "feature-extraction", "vl-embedding").
        trust_remote_code: Cho phép chạy code từ HuggingFace Hub.
        extra_load_kwargs: Keyword args bổ sung cho model loading.
        force: Ghi đè nếu model đã tồn tại trên R2.

    Returns:
        Dict chứa kết quả: model_name, files_count, total_size_mb, output_path.
    """
    import gc
    from pathlib import Path

    import torch
    from loguru import logger
    from transformers import (
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForSpeechSeq2Seq,
    )

    # AutoModelForImageTextToText chỉ có từ transformers >= 4.46
    # Converter image pin transformers < 4.46 cho DeepSeek-OCR-2 compatibility
    try:
        from transformers import AutoModelForImageTextToText as _AutoModelForImageTextToText
    except ImportError:
        _AutoModelForImageTextToText = None  # type: ignore[assignment]  # noqa: N806

    output_path = Path(MODELS_MOUNT_PATH) / model_name

    # Kiểm tra đã tồn tại chưa
    if output_path.exists() and any(output_path.iterdir()) and not force:
        existing_files = list(output_path.rglob("*"))
        existing_count = sum(1 for f in existing_files if f.is_file())
        logger.info(
            "Model {} đã tồn tại trên R2 ({} files). Bỏ qua. Dùng force=True để ghi đè.",
            model_name,
            existing_count,
        )
        return {
            "model_name": model_name,
            "status": "skipped",
            "reason": "already_exists",
            "files_count": existing_count,
        }

    # Xác định torch dtype
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    logger.info("Bắt đầu convert: {} (HF: {}, dtype: {})", model_name, hf_id, dtype)

    # Xác định model class
    model_class_map: dict[str, type | None] = {
        "AutoModel": AutoModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoModelForImageTextToText": _AutoModelForImageTextToText,
        "AutoModelForSpeechSeq2Seq": AutoModelForSpeechSeq2Seq,
    }
    cls = model_class_map.get(model_class)
    if cls is None:
        msg = (
            f"Model class '{model_class}' không khả dụng. "
            "Kiểm tra phiên bản transformers trong converter_image()."
        )
        raise ImportError(msg)

    # Build load kwargs
    load_kwargs: dict[str, object] = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "device_map": "cpu",
        **extra_load_kwargs,
    }

    # Download và load model
    logger.info("Đang tải model từ HuggingFace Hub...")
    model = cls.from_pretrained(hf_id, **load_kwargs)

    # Load tokenizer/processor
    logger.info("Đang tải tokenizer/processor...")
    vl_tasks = {"vl-embedding", "vl-reranker"}
    audio_tasks = {"automatic-speech-recognition"}

    if task in vl_tasks or task in audio_tasks:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            hf_id,
            trust_remote_code=trust_remote_code,
        )
    else:
        from transformers import AutoTokenizer

        processor = AutoTokenizer.from_pretrained(
            hf_id,
            trust_remote_code=trust_remote_code,
        )

    # Tạo thư mục output và lưu SafeTensors
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info("Đang lưu SafeTensors vào {}...", output_path)
    model.save_pretrained(str(output_path), safe_serialization=True)
    processor.save_pretrained(str(output_path))

    # Thống kê kết quả
    total_size = 0.0
    file_count = 0
    for f in sorted(output_path.rglob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / (1024**2)
            total_size += size_mb
            file_count += 1
            logger.info("  {} ({:.2f} MB)", f.name, size_mb)

    logger.info(
        "Convert {} hoàn tất: {} files, {:.2f} MB tổng cộng",
        model_name,
        file_count,
        total_size,
    )

    # Giải phóng bộ nhớ
    del model
    gc.collect()

    return {
        "model_name": model_name,
        "status": "success",
        "files_count": file_count,
        "total_size_mb": round(total_size, 2),
        "output_path": str(output_path),
    }
