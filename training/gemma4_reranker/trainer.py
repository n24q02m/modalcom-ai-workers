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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import MmRerankDataset, collate_fn
from .model import find_lm_head

if TYPE_CHECKING:
    from pathlib import Path

    from .config import StageConfig, TrainConfig


@dataclass
class TrainingContext:
    """Grouping of related training parameters and objects."""

    model: torch.nn.Module
    lm_head: torch.nn.Linear
    yes_id: int
    no_id: int
    config: TrainConfig
    optimizer: Any = None
    scheduler: Any = None
    scaler: Any = None
    mlflow_run: Any = None


def compute_loss(
    ctx: TrainingContext,
    inputs: dict,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute training loss with optional knowledge distillation.

    Uses hidden_states + lm_head weight to compute yes/no logits
    (more efficient than full vocab logits for 2-class problem).

    Args:
        ctx: Training context containing model, config, and IDs.
        inputs: Batch dict with input_ids, attention_mask, labels.

    Returns:
        (loss, metrics_dict) where metrics_dict has loss_ce, loss_kd, etc.
    """
    outputs = ctx.model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        output_hidden_states=True,
    )

    # Extract last hidden state at last non-padding position
    hidden = outputs.hidden_states[-1]  # (B, seq_len, hidden_dim)
    seq_lengths = inputs["attention_mask"].sum(dim=-1) - 1  # (B,)
    last_h = hidden[torch.arange(hidden.size(0)), seq_lengths]  # (B, hidden_dim)

    # Compute yes/no logits via lm_head weight
    logits = F.linear(last_h, ctx.lm_head.weight)  # (B, vocab_size)
    yes_logit = logits[:, ctx.yes_id]  # (B,)
    no_logit = logits[:, ctx.no_id]  # (B,)
    logits_yes_no = torch.stack([yes_logit, no_logit], dim=-1)  # (B, 2)

    labels = inputs["labels"]  # (B,) — 0=yes, 1=no

    # Cross-entropy loss
    loss_ce = F.cross_entropy(logits_yes_no, labels)

    metrics = {"loss_ce": loss_ce.item()}

    # Knowledge distillation (optional)
    kd_alpha = ctx.config.kd_alpha
    kd_temperature = ctx.config.kd_temperature

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


def train_one_epoch(
    ctx: TrainingContext,
    train_loader: DataLoader,
    global_step: int,
    epoch: int,
) -> tuple[int, float]:
    """Train for one epoch.

    Args:
        ctx: Training context.
        train_loader: Training DataLoader.
        global_step: Current global step counter.
        epoch: Current epoch number.

    Returns:
        (updated_global_step, avg_loss)
    """
    ctx.model.train()
    total_loss = 0.0
    num_batches = 0
    accum_loss = 0.0

    ctx.optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        # Move to GPU
        device = next(ctx.model.parameters()).device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # FP16 AMP forward
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss, metrics = compute_loss(ctx, batch)
            loss = loss / ctx.config.gradient_accumulation_steps

        ctx.scaler.scale(loss).backward()
        accum_loss += loss.item()

        # Gradient accumulation step
        if (batch_idx + 1) % ctx.config.gradient_accumulation_steps == 0:
            ctx.scaler.unscale_(ctx.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                ctx.model.parameters(), ctx.config.max_grad_norm
            )
            ctx.scaler.step(ctx.optimizer)
            ctx.scaler.update()
            ctx.scheduler.step()
            ctx.optimizer.zero_grad()

            global_step += 1
            total_loss += accum_loss * ctx.config.gradient_accumulation_steps
            num_batches += 1

            # Log metrics
            if global_step % ctx.config.logging_steps == 0:
                current_lr = ctx.scheduler.get_last_lr()[0]
                vram_alloc = torch.cuda.memory_allocated() / 1e9
                vram_reserved = torch.cuda.memory_reserved() / 1e9

                if ctx.mlflow_run:
                    _log_metrics(
                        ctx.mlflow_run,
                        {
                            "train/loss": accum_loss * ctx.config.gradient_accumulation_steps,
                            "train/loss_ce": metrics["loss_ce"],
                            "train/loss_kd": metrics["loss_kd"],
                            "train/lr": current_lr,
                            "train/grad_norm": grad_norm.item()
                            if isinstance(grad_norm, torch.Tensor)
                            else grad_norm,
                            "train/vram_allocated_gb": vram_alloc,
                            "train/vram_reserved_gb": vram_reserved,
                            "train/epoch": epoch,
                        },
                        step=global_step,
                    )

            accum_loss = 0.0

    avg_loss = total_loss / max(num_batches, 1)
    return global_step, avg_loss


@torch.no_grad()
def evaluate(
    ctx: TrainingContext,
    val_loader: DataLoader,
) -> float:
    """Evaluate model on validation set.

    Args:
        ctx: Training context.
        val_loader: Validation DataLoader.

    Returns:
        Average validation loss.
    """
    ctx.model.eval()
    total_loss = 0.0
    num_batches = 0

    device = next(ctx.model.parameters()).device

    for batch in val_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss, _ = compute_loss(ctx, batch)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def train_stage(
    model,
    processor,
    stage_config: StageConfig,
    train_config: TrainConfig,
    train_dataset: MmRerankDataset,
    val_dataset: MmRerankDataset,
) -> Path:
    """Run training for a single stage.

    Complete training pipeline:
    1. Setup optimizer, scheduler, scaler
    2. Train for num_epochs with gradient accumulation
    3. Evaluate every eval_steps, save best checkpoint
    4. Early stopping if no improvement
    5. Log everything to MLflow

    Args:
        model: PEFT model ready for training.
        processor: AutoProcessor for tokenization.
        stage_config: Stage-specific hyperparameters.
        train_config: Global training configuration.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.

    Returns:
        Path to best checkpoint directory.
    """
    import mlflow

    # Get yes/no token IDs
    yes_id = processor.tokenizer.convert_tokens_to_ids("yes")
    no_id = processor.tokenizer.convert_tokens_to_ids("no")
    lm_head = find_lm_head(model)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Kaggle: single worker to avoid memory issues
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.per_device_train_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    # Optimizer (paged AdamW 8-bit for VRAM efficiency)
    import bitsandbytes as bnb

    optimizer = bnb.optim.PagedAdamW8bit(
        model.parameters(),
        lr=stage_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    # LR scheduler
    total_steps = (
        len(train_loader) // train_config.gradient_accumulation_steps * stage_config.num_epochs
    )
    warmup_steps = int(total_steps * stage_config.warmup_ratio)

    from transformers import get_cosine_schedule_with_warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # FP16 GradScaler
    scaler = torch.amp.GradScaler("cuda")

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
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
        }
    )

    # Setup training context
    ctx = TrainingContext(
        model=model,
        lm_head=lm_head,
        yes_id=yes_id,
        no_id=no_id,
        config=train_config,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        mlflow_run=mlflow_run,
    )

    # Training loop
    global_step = 0
    best_val_loss = float("inf")
    patience_counter = 0
    best_ckpt_path = train_config.output_dir / f"stage{stage_config.stage.value}_best"

    for epoch in range(stage_config.num_epochs):
        start_time = time.time()
        global_step, train_loss = train_one_epoch(
            ctx=ctx,
            train_loader=train_loader,
            global_step=global_step,
            epoch=epoch,
        )
        epoch_time = time.time() - start_time

        # Evaluate
        val_loss = evaluate(ctx, val_loader)

        mlflow.log_metrics(
            {
                "eval/val_loss": val_loss,
                "train/epoch_loss": train_loss,
                "train/epoch_time_s": epoch_time,
            },
            step=global_step,
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_ckpt_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(best_ckpt_path))
            processor.save_pretrained(str(best_ckpt_path))
        else:
            patience_counter += 1

        # Save epoch checkpoint
        epoch_ckpt = train_config.output_dir / f"stage{stage_config.stage.value}_epoch{epoch}"
        epoch_ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(epoch_ckpt))

        # Early stopping
        if patience_counter >= stage_config.early_stopping_patience:
            mlflow.log_metric("train/early_stopped_epoch", epoch, step=global_step)
            break

    mlflow.log_metric("eval/best_val_loss", best_val_loss, step=global_step)
    mlflow.end_run()

    return best_ckpt_path


def _log_metrics(run, metrics: dict, step: int) -> None:
    """Log metrics to MLflow (safe import)."""
    import mlflow

    mlflow.log_metrics(metrics, step=step)
