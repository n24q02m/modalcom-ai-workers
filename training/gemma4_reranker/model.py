"""Model setup for QLoRA fine-tuning on T4.

Handles:
- BitsAndBytes NF4 quantization configuration
- LoRA adapter setup targeting decoder layers only
- Vision/Audio encoder freezing
- _find_lm_head() helper for training loss computation
- Gradient checkpointing configuration
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import LoraConfig as LoraConfigDC
from .config import QuantConfig


def make_bnb_config(quant_cfg: QuantConfig):
    """Create BitsAndBytesConfig from our QuantConfig dataclass.

    Args:
        quant_cfg: Quantization configuration.

    Returns:
        BitsAndBytesConfig for model loading.
    """
    from transformers import BitsAndBytesConfig

    compute_dtype = (
        torch.float16 if quant_cfg.bnb_4bit_compute_dtype == "float16" else torch.bfloat16
    )

    return BitsAndBytesConfig(
        load_in_4bit=quant_cfg.load_in_4bit,
        bnb_4bit_quant_type=quant_cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant_cfg.bnb_4bit_use_double_quant,
    )


def make_lora_config(lora_cfg: LoraConfigDC):
    """Create PEFT LoraConfig from our LoraConfig dataclass.

    Args:
        lora_cfg: LoRA adapter configuration.

    Returns:
        peft.LoraConfig for get_peft_model().
    """
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        target_modules=list(lora_cfg.target_modules),
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        task_type=TaskType.CAUSAL_LM,
    )


def load_model_for_training(
    model_id: str,
    quant_cfg: QuantConfig,
    lora_cfg: LoraConfigDC,
):
    """Load Gemma-4-E4B with QLoRA for fine-tuning on T4.

    Pipeline:
    1. Load base model with NF4 quantization
    2. Freeze vision + audio encoders
    3. Apply LoRA adapters to decoder layers
    4. Enable gradient checkpointing

    Args:
        model_id: HuggingFace model ID (e.g., "google/gemma-4-E4B-it").
        quant_cfg: BitsAndBytes quantization config.
        lora_cfg: LoRA adapter config.

    Returns:
        (model, processor) tuple ready for training.
    """
    from peft import get_peft_model
    from transformers import AutoProcessor, Gemma4ForConditionalGeneration

    bnb_config = make_bnb_config(quant_cfg)

    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # Freeze vision and audio encoders (only train decoder via LoRA)
    freeze_encoders(model)

    # Apply LoRA
    peft_config = make_lora_config(lora_cfg)
    model = get_peft_model(model, peft_config)
    model.enable_input_require_grads()

    # Enable gradient checkpointing (saves ~40% activation VRAM)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    return model, processor


def freeze_encoders(model) -> int:
    """Freeze vision and audio encoder parameters.

    Only decoder layers (targeted by LoRA) remain trainable.

    Args:
        model: Gemma4ForConditionalGeneration instance.

    Returns:
        Number of parameters frozen.
    """
    frozen_count = 0
    for name, param in model.named_parameters():
        if "vision" in name or "audio" in name:
            param.requires_grad = False
            frozen_count += param.numel()
    return frozen_count


def find_lm_head(model) -> nn.Linear:
    """Find the language model head (lm_head) weight matrix.

    Gemma4ForConditionalGeneration may have lm_head directly
    or nested under language_model sub-module.

    Used during training to compute logits from hidden states
    via F.linear(last_hidden, lm_head.weight) -- more efficient
    than full forward pass when we only need yes/no logits.

    Args:
        model: Gemma4ForConditionalGeneration or PeftModel.

    Returns:
        nn.Linear module containing the lm_head weights.

    Raises:
        AttributeError: If lm_head cannot be found.
    """
    # Unwrap PEFT model if needed
    base = model
    if hasattr(model, "base_model"):
        base = model.base_model
    if hasattr(base, "model"):
        base = base.model

    # Try direct access
    if hasattr(base, "lm_head") and isinstance(base.lm_head, nn.Linear):
        return base.lm_head

    # Try via language_model sub-module
    if hasattr(base, "language_model") and hasattr(base.language_model, "lm_head"):
        head = base.language_model.lm_head
        if isinstance(head, nn.Linear):
            return head

    # Fallback: search all modules
    for name, module in base.named_modules():
        if "lm_head" in name and isinstance(module, nn.Linear):
            return module

    head_attrs = [a for a in dir(base) if "head" in a.lower()]
    raise AttributeError(
        f"Cannot find lm_head in {type(model).__name__}. "
        f"Tried: model.lm_head, model.language_model.lm_head, named_modules. "
        f"Available attrs with 'head': {head_attrs}"
    )


def count_trainable_params(model) -> tuple[int, int]:
    """Count trainable and total parameters.

    Returns:
        (trainable_params, total_params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
