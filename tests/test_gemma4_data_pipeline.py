"""Tests for Gemma4 reranker data pipeline schema and conversion."""

from __future__ import annotations

import json

from training.gemma4_reranker.data_pipeline import (
    PointwiseSample,
    TrainSample,
    build_stage_dataset_from_grouped,
    deterministic_split_pointwise,
    grouped_to_pointwise,
    read_pointwise_jsonl,
    write_jsonl,
)


def test_grouped_to_pointwise_expansion() -> None:
    grouped = TrainSample(
        query="How to reset password?",
        positive="Open settings and choose reset.",
        negatives=["Banana is yellow.", "GPU drivers are outdated."],
        modality="text",
        language="en",
        source="unit-test",
        teacher_pos_score=0.92,
        teacher_neg_scores=[0.08, 0.12],
    )

    rows = grouped_to_pointwise(grouped)

    assert len(rows) == 3
    assert rows[0].label == 1
    assert rows[0].teacher_score == 0.92
    assert rows[1].label == 0
    assert rows[2].label == 0


def test_deterministic_split_is_query_group_aware() -> None:
    samples = [
        PointwiseSample(
            query="same query",
            document=f"doc {i}",
            label=1 if i == 0 else 0,
            modality="text",
            query_id="q-1",
        )
        for i in range(4)
    ]

    train, val = deterministic_split_pointwise(samples, val_ratio=0.5, seed=123)

    # All rows for the same query_id must stay in one split to avoid leakage
    assert not (train and val)


def test_build_stage_dataset_from_grouped(tmp_path) -> None:
    grouped_path = tmp_path / "grouped.jsonl"
    train_path = tmp_path / "stage1_train.jsonl"
    val_path = tmp_path / "stage1_val.jsonl"
    report_path = tmp_path / "stage1_report.json"

    grouped_rows = [
        TrainSample(
            query=f"query {i}",
            positive=f"positive {i}",
            negatives=[f"negative {i}-1", f"negative {i}-2"],
            modality="text",
            language="en" if i % 2 == 0 else "vi",
            source="synthetic",
            teacher_pos_score=0.9,
            teacher_neg_scores=[0.1, 0.2],
        )
        for i in range(6)
    ]

    write_jsonl(grouped_rows, grouped_path)

    report = build_stage_dataset_from_grouped(
        grouped_jsonl_path=grouped_path,
        train_out_path=train_path,
        val_out_path=val_path,
        val_ratio=0.2,
        seed=42,
        deduplicate=True,
        report_out_path=report_path,
    )

    assert report["input_grouped_count"] == 6
    assert report["expanded_pointwise_count"] == 18
    assert report["train_count"] + report["val_count"] == 18

    written = read_pointwise_jsonl(train_path, strict=True) + read_pointwise_jsonl(val_path, strict=True)
    assert len(written) == 18

    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved_report["train_summary"]["count"] == report["train_count"]
