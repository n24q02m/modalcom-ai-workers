"""Stage-based training entrypoint for Gemma-4 multilingual multimodal reranker.

This script is intentionally training-only. Serving is handled separately by
`src/ai_workers/workers/mm_reranker.py` via deployed merged checkpoints.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from .config import STAGE_CONFIGS, Stage, TrainConfig
from .dataset import MmRerankDataset
from .merge import merge_and_push
from .model import count_trainable_params, load_model_for_training
from .trainer import train_stage


def set_global_seed(seed: int) -> None:
    """Set all relevant RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_stage(raw: str) -> Stage:
    """Parse stage from '1|2|3' or 'stage1|stage2|stage3'."""
    value = raw.strip().lower()
    mapping = {
        "1": Stage.TEXT_IMAGE,
        "2": Stage.AUDIO,
        "3": Stage.VIDEO,
        "stage1": Stage.TEXT_IMAGE,
        "stage2": Stage.AUDIO,
        "stage3": Stage.VIDEO,
        "text_image": Stage.TEXT_IMAGE,
        "audio": Stage.AUDIO,
        "video": Stage.VIDEO,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported stage: {raw}")
    return mapping[value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Gemma-4 reranker (stage-based)")
    parser.add_argument("--stage", type=str, default="stage1", help="1|2|3 or stage1|stage2|stage3")
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        required=True,
        help="Pointwise train JSONL path",
    )
    parser.add_argument(
        "--val-jsonl",
        type=Path,
        required=True,
        help="Pointwise val JSONL path",
    )
    parser.add_argument(
        "--base-model-or-checkpoint",
        type=str,
        default="",
        help="Override base model/checkpoint for this stage",
    )
    parser.add_argument(
        "--hub-repo-id",
        type=str,
        default="",
        help="Override hub repo id for final merged model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/checkpoints"),
        help="Checkpoint output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Global random seed",
    )
    parser.add_argument(
        "--merge-after-train",
        action="store_true",
        help="Merge LoRA adapter after stage completes",
    )
    parser.add_argument(
        "--push-merged",
        action="store_true",
        help="Push merged model to hub when --merge-after-train is enabled",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage = parse_stage(args.stage)

    set_global_seed(args.seed)

    train_cfg = TrainConfig()
    train_cfg.output_dir = args.output_dir
    if args.hub_repo_id:
        train_cfg.hub_repo_id = args.hub_repo_id

    stage_cfg = STAGE_CONFIGS[stage]

    # For stage2/stage3, caller should pass previous stage checkpoint path
    if args.base_model_or_checkpoint:
        stage_cfg = replace(stage_cfg, base_model_or_checkpoint=args.base_model_or_checkpoint)

    if stage != Stage.TEXT_IMAGE and not stage_cfg.base_model_or_checkpoint:
        raise ValueError(
            "Stage 2/3 requires --base-model-or-checkpoint to point to previous stage checkpoint"
        )

    logger.info("Starting stage {} training", stage.value)
    logger.info("Base model/checkpoint: {}", stage_cfg.base_model_or_checkpoint)
    logger.info("Train JSONL: {}", args.train_jsonl)
    logger.info("Val JSONL: {}", args.val_jsonl)

    model, processor = load_model_for_training(
        model_id=stage_cfg.base_model_or_checkpoint,
        quant_cfg=train_cfg.quant,
        lora_cfg=train_cfg.lora,
    )

    trainable, total = count_trainable_params(model)
    logger.info(
        "Trainable params: {} / {} ({:.4f}%)",
        trainable,
        total,
        100.0 * trainable / max(total, 1),
    )

    train_dataset = MmRerankDataset(
        data_path=args.train_jsonl,
        processor=processor,
        config=train_cfg.data,
        reranker_prefix=train_cfg.reranker_prefix,
    )
    val_dataset = MmRerankDataset(
        data_path=args.val_jsonl,
        processor=processor,
        config=train_cfg.data,
        reranker_prefix=train_cfg.reranker_prefix,
    )

    best_ckpt = train_stage(
        model=model,
        processor=processor,
        stage_config=stage_cfg,
        train_config=train_cfg,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    logger.info("Best checkpoint: {}", best_ckpt)

    if args.merge_after_train:
        merged_path = train_cfg.output_dir / f"stage{stage.value}_merged"
        merge_and_push(
            adapter_path=best_ckpt,
            base_model_id=train_cfg.base_model_id,
            hub_repo_id=train_cfg.hub_repo_id,
            stage=stage.value,
            output_dir=merged_path,
            push=args.push_merged,
        )
        logger.info("Merged model ready at {}", merged_path)


if __name__ == "__main__":
    main()
