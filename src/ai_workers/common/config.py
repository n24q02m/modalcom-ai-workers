"""Model registry — single source of truth for all model configurations.

Every model's metadata (HuggingFace ID, precision, GPU, task, serving engine,
model class) is defined here. Both the CLI tools (convert, upload, deploy)
and the Modal workers reference this registry.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class Task(enum.StrEnum):
    """Supported model tasks."""

    EMBEDDING = "feature-extraction"
    RERANKER_LLM = "reranker-llm"
    VL_EMBEDDING = "vl-embedding"
    VL_RERANKER = "vl-reranker"
    OCR = "ocr"
    ASR = "automatic-speech-recognition"


class Precision(enum.StrEnum):
    """Weight precision for model storage and inference."""

    FP16 = "fp16"
    BF16 = "bf16"


class GPU(enum.StrEnum):
    """Modal GPU types."""

    T4 = "T4"
    A10G = "A10G"


class ServingEngine(enum.StrEnum):
    """How the model is served on Modal."""

    VLLM = "vllm"
    CUSTOM_FASTAPI = "custom-fastapi"


class Tier(enum.StrEnum):
    """Model size tier for routing."""

    LIGHT = "light"
    HEAVY = "heavy"


class ModelClassType(enum.StrEnum):
    """Transformers model class to use for loading."""

    AUTO_MODEL = "AutoModel"
    CAUSAL_LM = "AutoModelForCausalLM"
    IMAGE_TEXT_TO_TEXT = "AutoModelForImageTextToText"
    SEQ2SEQ = "AutoModelForSpeechSeq2Seq"


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a single model variant."""

    # Identity
    name: str  # Registry key, e.g. "qwen3-embedding-0.6b"
    hf_id: str  # HuggingFace model ID, e.g. "Qwen/Qwen3-Embedding-0.6B"
    task: Task
    tier: Tier

    # Inference
    precision: Precision = Precision.FP16
    gpu: GPU = GPU.T4
    serving_engine: ServingEngine = ServingEngine.CUSTOM_FASTAPI
    model_class: ModelClassType = ModelClassType.AUTO_MODEL

    # Transformers options
    trust_remote_code: bool = True

    # Modal app name (auto-derived from name if not set)
    modal_app_name: str = ""

    # Worker module path for `modal deploy`
    worker_module: str = ""

    # Modal app variable name within the worker module (for multi-app files)
    modal_app_var: str = ""

    # Extra kwargs for model loading
    extra_load_kwargs: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.modal_app_name:
            app_name = f"ai-workers-{self.name}"
            object.__setattr__(self, "modal_app_name", app_name)


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, ModelConfig] = {}


def _register(config: ModelConfig) -> ModelConfig:
    MODEL_REGISTRY[config.name] = config
    return config


# --- Qwen3 Embedding (merged: light + heavy on single A10G) ---
_register(
    ModelConfig(
        name="qwen3-embedding-0.6b",
        hf_id="Qwen/Qwen3-Embedding-0.6B",
        task=Task.EMBEDDING,
        tier=Tier.LIGHT,
        precision=Precision.FP16,
        gpu=GPU.A10G,
        serving_engine=ServingEngine.CUSTOM_FASTAPI,
        model_class=ModelClassType.AUTO_MODEL,
        worker_module="ai_workers.workers.embedding",
        modal_app_var="embedding_app",
        modal_app_name="ai-workers-embedding",
    )
)

_register(
    ModelConfig(
        name="qwen3-embedding-8b",
        hf_id="Qwen/Qwen3-Embedding-8B",
        task=Task.EMBEDDING,
        tier=Tier.HEAVY,
        precision=Precision.FP16,
        gpu=GPU.A10G,
        serving_engine=ServingEngine.CUSTOM_FASTAPI,
        model_class=ModelClassType.AUTO_MODEL,
        worker_module="ai_workers.workers.embedding",
        modal_app_var="embedding_app",
        modal_app_name="ai-workers-embedding",
    )
)

# --- Qwen3 Reranker (merged: light + heavy on single A10G) ---
_register(
    ModelConfig(
        name="qwen3-reranker-0.6b",
        hf_id="Qwen/Qwen3-Reranker-0.6B",
        task=Task.RERANKER_LLM,
        tier=Tier.LIGHT,
        precision=Precision.FP16,
        gpu=GPU.A10G,
        serving_engine=ServingEngine.CUSTOM_FASTAPI,
        model_class=ModelClassType.CAUSAL_LM,
        worker_module="ai_workers.workers.reranker",
        modal_app_var="reranker_app",
        modal_app_name="ai-workers-reranker",
    )
)

_register(
    ModelConfig(
        name="qwen3-reranker-8b",
        hf_id="Qwen/Qwen3-Reranker-8B",
        task=Task.RERANKER_LLM,
        tier=Tier.HEAVY,
        precision=Precision.FP16,
        gpu=GPU.A10G,
        serving_engine=ServingEngine.CUSTOM_FASTAPI,
        model_class=ModelClassType.CAUSAL_LM,
        worker_module="ai_workers.workers.reranker",
        modal_app_var="reranker_app",
        modal_app_name="ai-workers-reranker",
    )
)

# --- Qwen3 VL Embedding (merged: light + heavy on single A10G) ---
_register(
    ModelConfig(
        name="qwen3-vl-embedding-2b",
        hf_id="Qwen/Qwen3-VL-Embedding-2B",
        task=Task.VL_EMBEDDING,
        tier=Tier.LIGHT,
        precision=Precision.FP16,
        gpu=GPU.A10G,
        serving_engine=ServingEngine.CUSTOM_FASTAPI,
        model_class=ModelClassType.AUTO_MODEL,
        worker_module="ai_workers.workers.vl_embedding",
        modal_app_var="vl_embedding_app",
        modal_app_name="ai-workers-vl-embedding",
    )
)

_register(
    ModelConfig(
        name="qwen3-vl-embedding-8b",
        hf_id="Qwen/Qwen3-VL-Embedding-8B",
        task=Task.VL_EMBEDDING,
        tier=Tier.HEAVY,
        precision=Precision.FP16,
        gpu=GPU.A10G,
        serving_engine=ServingEngine.CUSTOM_FASTAPI,
        model_class=ModelClassType.AUTO_MODEL,
        worker_module="ai_workers.workers.vl_embedding",
        modal_app_var="vl_embedding_app",
        modal_app_name="ai-workers-vl-embedding",
    )
)

# --- Qwen3 VL Reranker (merged: light + heavy on single A10G) ---
_register(
    ModelConfig(
        name="qwen3-vl-reranker-2b",
        hf_id="Qwen/Qwen3-VL-Reranker-2B",
        task=Task.VL_RERANKER,
        tier=Tier.LIGHT,
        precision=Precision.FP16,
        gpu=GPU.A10G,
        serving_engine=ServingEngine.CUSTOM_FASTAPI,
        model_class=ModelClassType.IMAGE_TEXT_TO_TEXT,
        worker_module="ai_workers.workers.vl_reranker",
        modal_app_var="vl_reranker_app",
        modal_app_name="ai-workers-vl-reranker",
    )
)

_register(
    ModelConfig(
        name="qwen3-vl-reranker-8b",
        hf_id="Qwen/Qwen3-VL-Reranker-8B",
        task=Task.VL_RERANKER,
        tier=Tier.HEAVY,
        precision=Precision.FP16,
        gpu=GPU.A10G,
        serving_engine=ServingEngine.CUSTOM_FASTAPI,
        model_class=ModelClassType.IMAGE_TEXT_TO_TEXT,
        worker_module="ai_workers.workers.vl_reranker",
        modal_app_var="vl_reranker_app",
        modal_app_name="ai-workers-vl-reranker",
    )
)

# --- DeepSeek OCR v2 ---
_register(
    ModelConfig(
        name="deepseek-ocr-2",
        hf_id="deepseek-ai/DeepSeek-OCR-2",
        task=Task.OCR,
        tier=Tier.HEAVY,
        precision=Precision.BF16,  # BF16 required — trained in BF16, FP16 causes degradation
        gpu=GPU.A10G,  # A10G+ supports BF16 natively; T4 does NOT
        serving_engine=ServingEngine.CUSTOM_FASTAPI,
        model_class=ModelClassType.AUTO_MODEL,
        trust_remote_code=True,
        worker_module="ai_workers.workers.ocr",
        modal_app_var="ocr_app",
        extra_load_kwargs={"use_safetensors": True},
    )
)

# --- Whisper Large v3 ---
_register(
    ModelConfig(
        name="whisper-large-v3",
        hf_id="openai/whisper-large-v3",
        task=Task.ASR,
        tier=Tier.HEAVY,
        precision=Precision.FP16,
        gpu=GPU.T4,
        serving_engine=ServingEngine.CUSTOM_FASTAPI,
        model_class=ModelClassType.SEQ2SEQ,
        trust_remote_code=False,
        worker_module="ai_workers.workers.asr",
        modal_app_var="asr_app",
    )
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_model(name: str) -> ModelConfig:
    """Get a model config by registry name. Raises KeyError if not found."""
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        msg = f"Model '{name}' not found. Available: {available}"
        raise KeyError(msg)
    return MODEL_REGISTRY[name]


def list_models(*, task: Task | None = None, tier: Tier | None = None) -> list[ModelConfig]:
    """List models, optionally filtered by task and/or tier."""
    models = list(MODEL_REGISTRY.values())
    if task is not None:
        models = [m for m in models if m.task == task]
    if tier is not None:
        models = [m for m in models if m.tier == tier]
    return models


def get_torch_dtype(precision: Precision):
    """Convert Precision enum to torch dtype. Import torch lazily."""
    import torch

    return torch.float16 if precision == Precision.FP16 else torch.bfloat16


def get_model_class(model_class_type: ModelClassType):
    """Get the transformers model class. Import lazily."""
    from transformers import (
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoModelForSpeechSeq2Seq,
    )

    mapping = {
        ModelClassType.AUTO_MODEL: AutoModel,
        ModelClassType.CAUSAL_LM: AutoModelForCausalLM,
        ModelClassType.IMAGE_TEXT_TO_TEXT: AutoModelForImageTextToText,
        ModelClassType.SEQ2SEQ: AutoModelForSpeechSeq2Seq,
    }
    return mapping[model_class_type]
