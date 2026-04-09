"""Training configuration dataclasses.

All hyperparameters, paths, and stage-specific settings for the
3-stage Gemma-4-E4B reranker fine-tuning pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path


class Stage(IntEnum):
    """Training stages for incremental modality learning."""

    TEXT_IMAGE = 1
    AUDIO = 2
    VIDEO = 3


@dataclass(frozen=True)
class LoraConfig:
    """LoRA adapter configuration."""

    r: int = 16
    lora_alpha: int = 64  # alpha/r = 4
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass(frozen=True)
class QuantConfig:
    """BitsAndBytes NF4 quantization configuration for T4."""

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"  # T4 does NOT support native BF16
    bnb_4bit_use_double_quant: bool = True  # Save ~0.3GB VRAM


@dataclass(frozen=True)
class DataConfig:
    """Dataset paths and processing parameters."""

    # Data directory root (Kaggle: /kaggle/working/data)
    data_dir: Path = Path("/kaggle/working/data")

    # JSONL file paths per stage
    stage1_train: str = "stage1_train.jsonl"
    stage1_val: str = "stage1_val.jsonl"
    stage2_train: str = "stage2_train.jsonl"
    stage2_val: str = "stage2_val.jsonl"
    stage3_train: str = "stage3_train.jsonl"
    stage3_val: str = "stage3_val.jsonl"

    # Processing
    max_seq_length: int = 512
    visual_token_budget: int = 280
    video_tokens_per_frame: int = 70
    max_video_frames_train: int = 4
    max_video_frames_serve: int = 8
    max_audio_duration_s: float = 15.0  # Training: shorter audio for VRAM
    max_video_duration_s: float = 30.0  # Training: shorter video for VRAM

    # Hard negative mining
    num_negatives: int = 7
    skip_top_k: int = 10  # Skip top-10 (may be true positives)
    neg_pool_end: int = 50  # Sample from rank 10-50

    def train_path(self, stage: Stage) -> Path:
        """Get training data path for a stage."""
        mapping = {
            Stage.TEXT_IMAGE: self.stage1_train,
            Stage.AUDIO: self.stage2_train,
            Stage.VIDEO: self.stage3_train,
        }
        return self.data_dir / mapping[stage]

    def val_path(self, stage: Stage) -> Path:
        """Get validation data path for a stage."""
        mapping = {
            Stage.TEXT_IMAGE: self.stage1_val,
            Stage.AUDIO: self.stage2_val,
            Stage.VIDEO: self.stage3_val,
        }
        return self.data_dir / mapping[stage]


@dataclass(frozen=True)
class StageConfig:
    """Per-stage training hyperparameters."""

    stage: Stage
    base_model_or_checkpoint: str  # HF ID or path to previous stage checkpoint
    learning_rate: float
    num_epochs: int
    warmup_ratio: float
    replay_ratio: float = 0.0  # 0 for stage 1, 0.1 for stage 2/3
    eval_steps: int = 1000  # Evaluate every N steps
    early_stopping_patience: int = 3  # Stop after N evals without improvement


# Pre-defined stage configurations
STAGE_CONFIGS = {
    Stage.TEXT_IMAGE: StageConfig(
        stage=Stage.TEXT_IMAGE,
        base_model_or_checkpoint="google/gemma-4-E4B-it",
        learning_rate=2e-4,
        num_epochs=3,
        warmup_ratio=0.1,
        replay_ratio=0.0,
        eval_steps=1000,
    ),
    Stage.AUDIO: StageConfig(
        stage=Stage.AUDIO,
        base_model_or_checkpoint="",  # Set to stage 1 checkpoint path at runtime
        learning_rate=5e-5,
        num_epochs=2,
        warmup_ratio=0.05,
        replay_ratio=0.1,
        eval_steps=500,
    ),
    Stage.VIDEO: StageConfig(
        stage=Stage.VIDEO,
        base_model_or_checkpoint="",  # Set to stage 2 checkpoint path at runtime
        learning_rate=5e-5,
        num_epochs=2,
        warmup_ratio=0.05,
        replay_ratio=0.1,
        eval_steps=500,
    ),
}


@dataclass
class TrainConfig:
    """Top-level training configuration."""

    # Model
    base_model_id: str = "google/gemma-4-E4B-it"
    hub_repo_id: str = "n24q02m/gemma4-e4b-reranker-v1"

    # Hardware
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8  # Effective batch = 8
    gradient_checkpointing: bool = True
    fp16: bool = True  # T4 constraint
    bf16: bool = False

    # Optimizer
    optimizer: str = "paged_adamw_8bit"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"

    # Save & logging
    output_dir: Path = Path("/kaggle/working/checkpoints")
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    logging_steps: int = 10

    # Knowledge distillation
    kd_alpha: float = 0.3  # Blend: (1-alpha)*CE + alpha*KD
    kd_temperature: float = 2.0

    # Reranker prompt
    reranker_prefix: str = (
        'Judge whether the Document is relevant to the Query. '
        'Answer only "yes" or "no".'
    )

    # MLflow
    mlflow_tracking_uri: str = ""  # Set from env: MLFLOW_TRACKING_URI
    mlflow_experiment_name: str = "gemma4-reranker"

    # Sub-configs
    lora: LoraConfig = field(default_factory=LoraConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Replay
    replay_max_per_modality: int = 20  # 20 samples per modality type in buffer
