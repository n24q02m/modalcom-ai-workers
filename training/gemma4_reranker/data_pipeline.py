"""Data pipeline for production-grade multimodal reranker datasets.

This module defines 2 schema layers:
1) Grouped samples (`TrainSample`) from hard-negative mining
2) Pointwise samples (`PointwiseSample`) used directly by trainer loss

It also provides:
- strict JSONL validation
- grouped -> pointwise expansion
- deterministic split to avoid leakage
- deduplication and quality report generation
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


ALLOWED_MODALITIES = {"text", "image", "audio", "video"}


def _is_non_empty_text(value: str) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_score(name: str, score: float | None) -> None:
    if score is None:
        return
    if not (0.0 <= float(score) <= 1.0):
        raise ValueError(f"{name} must be in [0.0, 1.0], got: {score}")


def _safe_pick(values: list[Any], idx: int) -> Any | None:
    if not values:
        return None
    if idx < 0 or idx >= len(values):
        return None
    return values[idx]


@dataclass
class TrainSample:
    """Grouped training sample from hard-negative mining (Spec v5)."""

    query: str
    positive: str
    negatives: list[str]
    modality: str  # text, image, audio, video

    language: str = "und"
    source: str = "unknown"
    split: str = "train"

    query_id: str | None = None
    positive_id: str | None = None
    negative_ids: list[str | None] = field(default_factory=list)

    query_image: str | None = None
    query_audio: str | None = None
    query_video: str | None = None

    positive_image: str | None = None
    positive_audio: str | None = None
    positive_video: str | None = None

    negative_images: list[str | None] = field(default_factory=list)
    negative_audios: list[str | None] = field(default_factory=list)
    negative_videos: list[str | None] = field(default_factory=list)

    teacher_pos_score: float | None = None
    teacher_neg_scores: list[float] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate grouped sample integrity."""
        if self.modality not in ALLOWED_MODALITIES:
            raise ValueError(
                f"Invalid modality '{self.modality}'. Expected one of {sorted(ALLOWED_MODALITIES)}"
            )

        if not _is_non_empty_text(self.query):
            raise ValueError("query must be non-empty")
        if not _is_non_empty_text(self.positive):
            raise ValueError("positive must be non-empty")

        if not self.negatives:
            raise ValueError("negatives must be non-empty")
        if any(not _is_non_empty_text(n) for n in self.negatives):
            raise ValueError("all negatives must be non-empty strings")

        if self.negative_ids and len(self.negative_ids) != len(self.negatives):
            raise ValueError("negative_ids length must equal negatives length")

        for field_name, values in (
            ("negative_images", self.negative_images),
            ("negative_audios", self.negative_audios),
            ("negative_videos", self.negative_videos),
        ):
            if values and len(values) != len(self.negatives):
                raise ValueError(f"{field_name} length must equal negatives length")

        _validate_score("teacher_pos_score", self.teacher_pos_score)
        if self.teacher_neg_scores and len(self.teacher_neg_scores) != len(self.negatives):
            raise ValueError("teacher_neg_scores length must equal negatives length")
        for idx, score in enumerate(self.teacher_neg_scores):
            _validate_score(f"teacher_neg_scores[{idx}]", score)

    def to_dict(self) -> dict[str, Any]:
        """Serialize grouped sample to JSON-compatible dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainSample:
        """Deserialize grouped sample from dict (drops unknown keys)."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PointwiseSample:
    """Pointwise training sample consumed by training loss."""

    query: str
    document: str
    label: int  # 1=relevant, 0=irrelevant
    modality: str

    language: str = "und"
    source: str = "unknown"
    split: str = "train"

    query_id: str | None = None
    document_id: str | None = None

    query_image: str | None = None
    query_audio: str | None = None
    query_video: str | None = None

    document_image: str | None = None
    document_audio: str | None = None
    document_video: str | None = None

    teacher_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate pointwise sample integrity."""
        if self.modality not in ALLOWED_MODALITIES:
            raise ValueError(
                f"Invalid modality '{self.modality}'. Expected one of {sorted(ALLOWED_MODALITIES)}"
            )
        if self.label not in (0, 1):
            raise ValueError(f"label must be 0 or 1, got: {self.label}")
        if not _is_non_empty_text(self.query):
            raise ValueError("query must be non-empty")
        if not _is_non_empty_text(self.document):
            raise ValueError("document must be non-empty")
        _validate_score("teacher_score", self.teacher_score)

    def to_dict(self) -> dict[str, Any]:
        """Serialize pointwise sample to JSON-compatible dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PointwiseSample:
        """Deserialize pointwise sample from dict (drops unknown keys)."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def write_jsonl(samples: list[TrainSample | PointwiseSample], path: Path) -> int:
    """Write dataclass samples to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            payload = sample.to_dict()
            json.dump(payload, f, ensure_ascii=False)
            f.write("\n")
            count += 1
    return count


def read_jsonl(path: Path, strict: bool = True) -> list[TrainSample]:
    """Read grouped TrainSample JSONL."""
    samples: list[TrainSample] = []
    with open(path, encoding="utf-8") as f:
        for _line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            sample = TrainSample.from_dict(json.loads(line))
            if strict:
                sample.validate()
            samples.append(sample)
    return samples


def read_pointwise_jsonl(path: Path, strict: bool = True) -> list[PointwiseSample]:
    """Read pointwise PointwiseSample JSONL."""
    samples: list[PointwiseSample] = []
    with open(path, encoding="utf-8") as f:
        for _line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            sample = PointwiseSample.from_dict(json.loads(line))
            if strict:
                sample.validate()
            samples.append(sample)
    return samples


def grouped_to_pointwise(sample: TrainSample) -> list[PointwiseSample]:
    """Expand one grouped sample into 1 positive + N negative pointwise samples."""
    sample.validate()
    base_metadata = dict(sample.metadata)

    records: list[PointwiseSample] = [
        PointwiseSample(
            query=sample.query,
            document=sample.positive,
            label=1,
            modality=sample.modality,
            language=sample.language,
            source=sample.source,
            split=sample.split,
            query_id=sample.query_id,
            document_id=sample.positive_id,
            query_image=sample.query_image,
            query_audio=sample.query_audio,
            query_video=sample.query_video,
            document_image=sample.positive_image,
            document_audio=sample.positive_audio,
            document_video=sample.positive_video,
            teacher_score=sample.teacher_pos_score,
            metadata={**base_metadata, "candidate_type": "positive", "candidate_rank": 0},
        )
    ]

    for idx, negative in enumerate(sample.negatives):
        records.append(
            PointwiseSample(
                query=sample.query,
                document=negative,
                label=0,
                modality=sample.modality,
                language=sample.language,
                source=sample.source,
                split=sample.split,
                query_id=sample.query_id,
                document_id=_safe_pick(sample.negative_ids, idx),
                query_image=sample.query_image,
                query_audio=sample.query_audio,
                query_video=sample.query_video,
                document_image=_safe_pick(sample.negative_images, idx),
                document_audio=_safe_pick(sample.negative_audios, idx),
                document_video=_safe_pick(sample.negative_videos, idx),
                teacher_score=_safe_pick(sample.teacher_neg_scores, idx),
                metadata={
                    **base_metadata,
                    "candidate_type": "negative",
                    "candidate_rank": idx + 1,
                },
            )
        )

    for item in records:
        item.validate()
    return records


def expand_grouped_samples(samples: list[TrainSample]) -> list[PointwiseSample]:
    """Expand all grouped samples into pointwise samples."""
    expanded: list[PointwiseSample] = []
    for sample in samples:
        expanded.extend(grouped_to_pointwise(sample))
    return expanded


def deduplicate_pointwise(samples: list[PointwiseSample]) -> list[PointwiseSample]:
    """Deduplicate pointwise samples while preserving best teacher score coverage."""
    by_key: dict[tuple[str, str, int, str, str], PointwiseSample] = {}

    for sample in samples:
        key = (
            sample.query.strip(),
            sample.document.strip(),
            sample.label,
            sample.modality,
            sample.language,
        )
        prev = by_key.get(key)
        if prev is None:
            by_key[key] = sample
            continue

        # Prefer sample with teacher score if previous missing
        if prev.teacher_score is None and sample.teacher_score is not None:
            by_key[key] = sample

    return list(by_key.values())


def _split_group_key(sample: PointwiseSample) -> str:
    if sample.query_id:
        return sample.query_id
    digest = hashlib.sha1(sample.query.encode("utf-8")).hexdigest()
    return f"query::{digest}"


def deterministic_split_pointwise(
    samples: list[PointwiseSample],
    val_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[list[PointwiseSample], list[PointwiseSample]]:
    """Deterministically split data while keeping the same query in one split.

    This avoids retrieval leakage where the same query appears in train and val.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got: {val_ratio}")

    train: list[PointwiseSample] = []
    val: list[PointwiseSample] = []

    for sample in samples:
        group_key = _split_group_key(sample)
        bucket_seed = f"{seed}:{group_key}"
        bucket = int(hashlib.sha1(bucket_seed.encode("utf-8")).hexdigest()[:8], 16)
        threshold = int(val_ratio * 0xFFFFFFFF)

        if bucket <= threshold:
            val.append(sample)
        else:
            train.append(sample)

    # Safety fallback for tiny datasets while preserving query-group boundaries.
    # If everything lands in val, move the entire val split to train.
    # Do not force-pop single samples between train/val as it can cause leakage.
    if not train and val:
        train = list(val)
        val = []

    return train, val


def split_train_val(
    samples: list[PointwiseSample],
    val_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[list[PointwiseSample], list[PointwiseSample]]:
    """Backward-compatible split wrapper (deterministic query-aware split)."""
    return deterministic_split_pointwise(samples=samples, val_ratio=val_ratio, seed=seed)


def summarize_pointwise(samples: list[PointwiseSample]) -> dict[str, Any]:
    """Build a compact data quality report for pointwise samples."""
    label_counter = Counter(s.label for s in samples)
    modality_counter = Counter(s.modality for s in samples)
    language_counter = Counter(s.language for s in samples)
    source_counter = Counter(s.source for s in samples)

    teacher_scores = [s.teacher_score for s in samples if s.teacher_score is not None]
    doc_lens = [len(s.document) for s in samples]
    query_lens = [len(s.query) for s in samples]

    return {
        "count": len(samples),
        "labels": {str(k): v for k, v in sorted(label_counter.items())},
        "modalities": dict(sorted(modality_counter.items())),
        "languages": dict(sorted(language_counter.items())),
        "sources": dict(sorted(source_counter.items())),
        "teacher_coverage_ratio": (len(teacher_scores) / len(samples) if samples else 0.0),
        "teacher_score_mean": statistics.mean(teacher_scores) if teacher_scores else None,
        "teacher_score_min": min(teacher_scores) if teacher_scores else None,
        "teacher_score_max": max(teacher_scores) if teacher_scores else None,
        "query_len_chars_p50": int(statistics.median(query_lens)) if query_lens else 0,
        "doc_len_chars_p50": int(statistics.median(doc_lens)) if doc_lens else 0,
    }


def build_stage_dataset_from_grouped(
    grouped_jsonl_path: Path,
    train_out_path: Path,
    val_out_path: Path,
    val_ratio: float = 0.05,
    seed: int = 42,
    deduplicate: bool = True,
    report_out_path: Path | None = None,
) -> dict[str, Any]:
    """Convert grouped hard-negative JSONL into train/val pointwise JSONL.

    Returns a quality report dictionary and optionally writes it to disk.
    """
    grouped = read_jsonl(grouped_jsonl_path, strict=True)
    pointwise = expand_grouped_samples(grouped)

    if deduplicate:
        pointwise = deduplicate_pointwise(pointwise)

    train_samples, val_samples = deterministic_split_pointwise(
        pointwise,
        val_ratio=val_ratio,
        seed=seed,
    )

    train_count = write_jsonl(train_samples, train_out_path)
    val_count = write_jsonl(val_samples, val_out_path)

    report = {
        "input_grouped_count": len(grouped),
        "expanded_pointwise_count": len(pointwise),
        "train_count": train_count,
        "val_count": val_count,
        "val_ratio_target": val_ratio,
        "val_ratio_actual": val_count / max(train_count + val_count, 1),
        "train_summary": summarize_pointwise(train_samples),
        "val_summary": summarize_pointwise(val_samples),
    }

    if report_out_path is not None:
        report_out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    return report
