"""Prepare pointwise train/val JSONL from grouped hard-negative JSONL.

Usage:
    python -m training.gemma4_reranker.prepare_dataset \
      --input-grouped-jsonl /kaggle/working/stage1_train.jsonl \
      --output-dir /kaggle/working/data \
      --stage stage1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger

from .data_pipeline import build_stage_dataset_from_grouped

STAGE_TO_FILENAMES = {
    "stage1": ("stage1_train.jsonl", "stage1_val.jsonl", "stage1_report.json"),
    "stage2": ("stage2_train.jsonl", "stage2_val.jsonl", "stage2_report.json"),
    "stage3": ("stage3_train.jsonl", "stage3_val.jsonl", "stage3_report.json"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build production-ready reranker train/val JSONL")
    parser.add_argument(
        "--input-grouped-jsonl",
        type=Path,
        required=True,
        help="Input grouped JSONL path from hard-negative mining",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/data"),
        help="Directory to write output train/val JSONL",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=sorted(STAGE_TO_FILENAMES.keys()),
        default="stage1",
        help="Stage identifier for output filenames",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic split seed",
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Disable pointwise deduplication",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_name, val_name, report_name = STAGE_TO_FILENAMES[args.stage]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.output_dir / train_name
    val_path = args.output_dir / val_name
    report_path = args.output_dir / report_name

    logger.info("Preparing dataset for {}", args.stage)
    logger.info("Input grouped JSONL: {}", args.input_grouped_jsonl)
    logger.info("Output train JSONL: {}", train_path)
    logger.info("Output val JSONL: {}", val_path)

    report = build_stage_dataset_from_grouped(
        grouped_jsonl_path=args.input_grouped_jsonl,
        train_out_path=train_path,
        val_out_path=val_path,
        val_ratio=args.val_ratio,
        seed=args.seed,
        deduplicate=not args.no_deduplicate,
        report_out_path=report_path,
    )

    logger.info("Dataset preparation complete")
    logger.info("Train samples: {} | Val samples: {}", report["train_count"], report["val_count"])
    logger.info("Report written to {}", report_path)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
