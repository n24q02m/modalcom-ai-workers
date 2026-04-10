"""Data pipeline for preparing training JSONL files.

Converts raw datasets into the unified Grouped JSONL format
required for Knowledge Distillation and Cross-Entropy loss.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class TrainSample:
    """A single grouped training sample in the unified schema (Spec v5)."""

    query: str
    positive: str
    negatives: list[str]
    modality: str  # text, image, audio, video

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

    def to_dict(self) -> dict:
        """Serialize to dict, automatically matching the schema."""
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> TrainSample:
        """Deserialize from dict."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def write_jsonl(samples: list[TrainSample], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            json.dump(sample.to_dict(), f, ensure_ascii=False)
            f.write("\n")
    return len(samples)

def split_train_val(
    samples: list[TrainSample],
    val_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[list[TrainSample], list[TrainSample]]:
    import random
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_ratio))
    return shuffled[val_size:], shuffled[:val_size]
