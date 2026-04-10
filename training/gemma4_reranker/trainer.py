"""Training loop with FP16, MLflow logging, and early stopping.

Implements the core training logic for all 3 stages:
- Pointwise cross-entropy loss on yes/no logits
- Optional knowledge distillation from teacher scores
- MLflow metric/artifact logging
- VRAM monitoring
- Early stopping based on validation loss
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader

from .dataset import collate_fn
from .model import find_lm_head

if TYPE_CHECKING:
    from pathlib import Path

    from .config import StageConfig, TrainConfig
    from .dataset import MmRerankDataset


@dataclass
class TrainingAssets:
    """Group model, processor and datasets for cleaner signatures."""

    model: Any
    processor: Any
    train_dataset: MmRerankDataset
    val_dataset: MmRerankDataset


def compute_loss(
    model,
    inputs: dict,
    yes_id: int,
    no_id: int,
    lm_head: torch.nn.Linear,
    kd_alpha: float = 0.0,
    kd_temperature: float = 2.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute training loss with optional knowledge distillation.

    Uses hidden_states + lm_head weight to compute yes/no logits
    (more efficient than full vocab logits for 2-class problem).

    Args:
        model: Gemma4ForConditionalGeneration (PEFT wrapped).
        inputs: Batch dict with input_ids, attention_mask, labels.
        yes_id: Token ID for "yes".
        no_id: Token ID for "no".
        lm_head: lm_head Linear module for logit computation.
        kd_alpha: Knowledge distillation blend weight (0 = no KD).
        kd_temperature: KD softmax temperature.

    Returns:
        (loss, metrics_dict) where metrics_dict has loss_ce, loss_kd, etc.
    """
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        output_hidden_states=True,
    )

    # Extract last hidden state at last non-padding position
    hidden = outputs.hidden_states[-1]  # (B, seq_len, hidden_dim)
    seq_lengths = inputs["attention_mask"].sum(dim=-1) - 1  # (B,)
    last_h = hidden[torch.arange(hidden.size(0)), seq_lengths]  # (B, hidden_dim)

    # Compute yes/no logits via lm_head weight
    logits = F.linear(last_h, lm_head.weight)  # (B, vocab_size)
    yes_logit = logits[:, yes_id]  # (B,)
    no_logit = logits[:, no_id]  # (B,)
    logits_yes_no = torch.stack([yes_logit, no_logit], dim=-1)  # (B, 2)

    labels = inputs["labels"]  # (B,) — 0=yes, 1=no

    # Cross-entropy loss
    loss_ce = F.cross_entropy(logits_yes_no, labels)

    metrics = {"loss_ce": loss_ce.item()}

    # Knowledge distillation (optional)
    if kd_alpha > 0 and "teacher_scores" in inputs:
        teacher_scores = inputs["teacher_scores"]  # (B,) cosine sim from teacher
        # Convert teacher scores to [yes_prob, no_prob]
        teacher_yes = teacher_scores
        teacher_no = 1.0 - teacher_scores
        teacher_logits = torch.stack([teacher_yes, teacher_no], dim=-1)  # (B, 2)

        teacher_probs = F.softmax(teacher_logits / kd_temperature, dim=-1)
        student_log_probs = F.log_softmax(logits_yes_no / kd_temperature, dim=-1)
        loss_kd = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

        loss = (1 - kd_alpha) * loss_ce + kd_alpha * loss_kd * (kd_temperature**2)
        metrics["loss_kd"] = loss_kd.item()
    else:
        loss = loss_ce
        metrics["loss_kd"] = 0.0

    metrics["loss_total"] = loss.item()
    return loss, metrics


class RerankTrainer:
    """Trainer class to manage training state and logic."""

    def __init__(
        self,
        assets: TrainingAssets,
        stage_config: StageConfig,
        train_config: TrainConfig,
    ):
        self.assets = assets
        self.stage_config = stage_config
        self.train_config = train_config

        # Get yes/no token IDs
        self.yes_id = assets.processor.tokenizer.convert_tokens_to_ids("yes")
        self.no_id = assets.processor.tokenizer.convert_tokens_to_ids("no")
        self.lm_head = find_lm_head(assets.model)

        # DataLoaders
        self.train_loader = DataLoader(
            assets.train_dataset,
            batch_size=train_config.per_device_train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            assets.val_dataset,
            batch_size=train_config.per_device_train_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True,
        )

        # Optimizer (paged AdamW 8-bit for VRAM efficiency)
        import bitsandbytes as bnb

        self.optimizer = bnb.optim.PagedAdamW8bit(
            assets.model.parameters(),
            lr=stage_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )

        # LR scheduler
        total_steps = (
            len(self.train_loader)
            // train_config.gradient_accumulation_steps
            * stage_config.num_epochs
        )
        warmup_steps = int(total_steps * stage_config.warmup_ratio)

        from transformers import get_cosine_schedule_with_warmup

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # FP16 GradScaler
        self.scaler = torch.amp.GradScaler("cuda")
        self.global_step = 0

    def train_one_epoch(self, epoch: int, mlflow_run=None) -> float:
        """Train for one epoch."""
        self.assets.model.train()
        total_loss = 0.0
        num_batches = 0
        accum_loss = 0.0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            device = next(self.assets.model.parameters()).device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }

            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss, metrics = compute_loss(
                    self.assets.model,
                    batch,
                    yes_id=self.yes_id,
                    no_id=self.no_id,
                    lm_head=self.lm_head,
                    kd_alpha=self.train_config.kd_alpha,
                    kd_temperature=self.train_config.kd_temperature,
                )
                loss = loss / self.train_config.gradient_accumulation_steps

            self.scaler.scale(loss).backward()
            accum_loss += loss.item()

            if (batch_idx + 1) % self.train_config.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.assets.model.parameters(), self.train_config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1
                total_loss += accum_loss * self.train_config.gradient_accumulation_steps
                num_batches += 1

                if self.global_step % self.train_config.logging_steps == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    vram_alloc = torch.cuda.memory_allocated() / 1e9
                    vram_reserved = torch.cuda.memory_reserved() / 1e9

                    if mlflow_run:
                        _log_metrics(
                            mlflow_run,
                            {
                                "train/loss": accum_loss
                                * self.train_config.gradient_accumulation_steps,
                                "train/loss_ce": metrics["loss_ce"],
                                "train/loss_kd": metrics["loss_kd"],
                                "train/lr": current_lr,
                                "train/grad_norm": (
                                    grad_norm.item()
                                    if isinstance(grad_norm, torch.Tensor)
                                    else grad_norm
                                ),
                                "train/vram_allocated_gb": vram_alloc,
                                "train/vram_reserved_gb": vram_reserved,
                                "train/epoch": epoch,
                            },
                            step=self.global_step,
                        )

                accum_loss = 0.0

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate model on validation set."""
        self.assets.model.eval()
        total_loss = 0.0
        num_batches = 0

        device = next(self.assets.model.parameters()).device

        for batch in self.val_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }

            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss, _ = compute_loss(
                    self.assets.model,
                    batch,
                    yes_id=self.yes_id,
                    no_id=self.no_id,
                    lm_head=self.lm_head,
                )

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)


def train_stage(
    assets: TrainingAssets,
    stage_config: StageConfig,
    train_config: TrainConfig,
) -> Path:
    """Run training for a single stage."""
    import mlflow

    trainer = RerankTrainer(assets, stage_config, train_config)

    # MLflow setup
    if train_config.mlflow_tracking_uri:
        mlflow.set_tracking_uri(train_config.mlflow_tracking_uri)
    mlflow.set_experiment(train_config.mlflow_experiment_name)
    mlflow_run = mlflow.start_run(
        run_name=f"stage-{stage_config.stage.value}",
        tags={
            "stage": str(stage_config.stage.value),
            "model": train_config.base_model_id,
            "hardware": "kaggle-t4-16gb",
        },
    )

    # Log config
    mlflow.log_params(
        {
            "stage": stage_config.stage.value,
            "learning_rate": stage_config.learning_rate,
            "num_epochs": stage_config.num_epochs,
            "effective_batch_size": (
                train_config.per_device_train_batch_size * train_config.gradient_accumulation_steps
            ),
            "max_seq_length": train_config.data.max_seq_length,
            "lora_r": train_config.lora.r,
            "lora_alpha": train_config.lora.lora_alpha,
            "replay_ratio": stage_config.replay_ratio,
            "train_samples": len(assets.train_dataset),
            "val_samples": len(assets.val_dataset),
        }
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_ckpt_path = train_config.output_dir / f"stage{stage_config.stage.value}_best"

    for epoch in range(stage_config.num_epochs):
        start_time = time.time()
        train_loss = trainer.train_one_epoch(epoch=epoch, mlflow_run=mlflow_run)
        epoch_time = time.time() - start_time

        val_loss = trainer.evaluate()

        mlflow.log_metrics(
            {
                "eval/val_loss": val_loss,
                "train/epoch_loss": train_loss,
                "train/epoch_time_s": epoch_time,
            },
            step=trainer.global_step,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_ckpt_path.mkdir(parents=True, exist_ok=True)
            assets.model.save_pretrained(str(best_ckpt_path))
            assets.processor.save_pretrained(str(best_ckpt_path))
        else:
            patience_counter += 1

        epoch_ckpt = train_config.output_dir / f"stage{stage_config.stage.value}_epoch{epoch}"
        epoch_ckpt.mkdir(parents=True, exist_ok=True)
        assets.model.save_pretrained(str(epoch_ckpt))

        if patience_counter >= stage_config.early_stopping_patience:
            mlflow.log_metric("train/early_stopped_epoch", epoch, step=trainer.global_step)
            break

    mlflow.log_metric("eval/best_val_loss", best_val_loss, step=trainer.global_step)
    mlflow.end_run()

    return best_ckpt_path


def _log_metrics(run, metrics: dict, step: int) -> None:
    """Log metrics to MLflow (safe import)."""
    import mlflow

    mlflow.log_metrics(metrics, step=step)
