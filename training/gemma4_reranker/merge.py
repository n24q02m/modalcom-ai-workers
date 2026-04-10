"""LoRA merge and push pipeline.

Post-training pipeline:
1. Load base model (FP16) + LoRA adapter
2. merge_and_unload() -> full model
3. Cast to BF16 (serving precision on A10G)
4. save_pretrained with safetensors format
5. Push to HuggingFace Hub
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MergeConfig:
    """Configuration for LoRA merge and push.

    Attributes:
        adapter_path: Path to LoRA adapter directory.
        base_model_id: HuggingFace ID of the base model.
        hub_repo_id: Target HuggingFace Hub repo ID.
        stage: Training stage number (for commit message).
        output_dir: Local directory for merged model.
        push: Whether to push to HF Hub.
    """

    adapter_path: str | Path
    base_model_id: str
    hub_repo_id: str
    stage: int
    output_dir: str | Path = "./merged_model"
    push: bool = True


def merge_and_push(config: MergeConfig) -> Path:
    """Merge LoRA adapters with base model and optionally push to HF Hub.

    Pipeline:
    1. Load base model in FP16 on CPU (no GPU needed)
    2. Load LoRA adapter
    3. merge_and_unload() -> merged full-precision model
    4. Cast to BF16 (A10G native precision for serving)
    5. Save with safetensors format
    6. Push to HuggingFace Hub

    Args:
        config: Merge and push configuration.

    Returns:
        Path to the merged model directory.
    """
    import torch
    from loguru import logger
    from peft import PeftModel
    from transformers import AutoProcessor, Gemma4ForConditionalGeneration

    adapter_path = Path(config.adapter_path)
    output_dir = Path(config.output_dir)

    logger.info("Loading base model {} in FP16 on CPU...", config.base_model_id)
    base_model = Gemma4ForConditionalGeneration.from_pretrained(
        config.base_model_id,
        torch_dtype=torch.float16,
        device_map="cpu",  # Merge on CPU (no GPU required)
        trust_remote_code=True,
    )

    logger.info("Loading LoRA adapter from {}...", adapter_path)
    model = PeftModel.from_pretrained(base_model, str(adapter_path))

    logger.info("Merging LoRA adapters...")
    model = model.merge_and_unload()

    logger.info("Casting to BF16 (serving precision)...")
    model = model.to(torch.bfloat16)

    logger.info("Saving merged model to {}...", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        str(output_dir),
        safe_serialization=True,  # Use safetensors format
    )

    # Save processor alongside model
    logger.info("Saving processor...")
    processor = AutoProcessor.from_pretrained(config.base_model_id, trust_remote_code=True)
    processor.save_pretrained(str(output_dir))

    if config.push:
        logger.info("Pushing to HuggingFace Hub: {}...", config.hub_repo_id)
        model.push_to_hub(
            config.hub_repo_id,
            commit_message=f"feat: stage {config.stage} merged checkpoint",
            safe_serialization=True,
        )
        processor.push_to_hub(config.hub_repo_id)
        logger.info("Push complete!")

    return output_dir


def verify_merged_model(model_path: str | Path) -> dict:
    """Verify a merged model can be loaded and produces outputs.

    Sanity check: load model, run a dummy forward pass,
    verify yes/no token IDs exist.

    Args:
        model_path: Path to merged model directory.

    Returns:
        Dict with verification results (dtype, device, vocab_size, etc.)
    """
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_path = str(model_path)

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()

    # Check yes/no token IDs
    yes_id = processor.tokenizer.convert_tokens_to_ids("yes")
    no_id = processor.tokenizer.convert_tokens_to_ids("no")

    # Quick forward pass
    text = "Test input for verification."
    inputs = processor(text=text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    return {
        "dtype": str(next(model.parameters()).dtype),
        "vocab_size": outputs.logits.shape[-1],
        "yes_token_id": yes_id,
        "no_token_id": no_id,
        "logits_shape": list(outputs.logits.shape),
        "status": "ok",
    }
