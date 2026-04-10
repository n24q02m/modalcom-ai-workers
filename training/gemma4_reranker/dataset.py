"""PyTorch dataset and collate for multilingual multimodal reranker training.

This dataset consumes pointwise JSONL records produced by
`data_pipeline.build_stage_dataset_from_grouped`.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import torch
from loguru import logger
from torch.utils.data import Dataset

from .data_pipeline import PointwiseSample, read_pointwise_jsonl

if TYPE_CHECKING:
    from pathlib import Path

    from .config import DataConfig


class MmRerankDataset(Dataset):
    """Pointwise dataset for multimodal reranker training.

    Output labels use the trainer convention:
    - class 0 -> "yes" (relevant)
    - class 1 -> "no" (irrelevant)
    """

    def __init__(
        self,
        data_path: Path,
        processor,
        config: DataConfig,
        reranker_prefix: str,
    ) -> None:
        self.samples = read_pointwise_jsonl(data_path, strict=True)
        self.processor = processor
        self.config = config
        self.reranker_prefix = reranker_prefix

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        return self._process_sample(sample)

    def _process_sample(self, sample: PointwiseSample) -> dict:
        """Tokenize one pointwise sample into model-ready tensors."""
        content_parts: list[dict] = []
        images: list = []
        audios: list = []

        # Query-side media first
        if sample.query_image:
            img = self._load_image(sample.query_image)
            if img is not None:
                content_parts.append({"type": "image", "image": img})
                images.append(img)

        if sample.query_audio:
            audio_data = self._load_audio(sample.query_audio)
            if audio_data is not None:
                content_parts.append({"type": "audio", "audio": audio_data})
                audios.append(audio_data)

        if sample.query_video:
            for frame in self._load_video_frames(sample.query_video):
                content_parts.append({"type": "image", "image": frame})
                images.append(frame)

        # Text query payload
        content_parts.append({"type": "text", "text": f"<Query>\n{sample.query}\n</Query>"})

        # Document-side media
        if sample.document_image:
            img = self._load_image(sample.document_image)
            if img is not None:
                content_parts.append({"type": "image", "image": img})
                images.append(img)

        if sample.document_audio:
            audio_data = self._load_audio(sample.document_audio)
            if audio_data is not None:
                content_parts.append({"type": "audio", "audio": audio_data})
                audios.append(audio_data)

        if sample.document_video:
            for frame in self._load_video_frames(sample.document_video):
                content_parts.append({"type": "image", "image": frame})
                images.append(frame)

        content_parts.append(
            {"type": "text", "text": f"<Document>\n{sample.document}\n</Document>"}
        )

        messages = [
            {"role": "system", "content": self.reranker_prefix},
            {"role": "user", "content": content_parts},
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        proc_kwargs = {
            "text": text,
            "return_tensors": "pt",
            "padding": False,
            "truncation": True,
            "max_length": self.config.max_seq_length,
        }
        if images:
            proc_kwargs["images"] = images
        if audios:
            proc_kwargs["audios"] = audios

        inputs = self.processor(**proc_kwargs)

        # Convert to single-sample tensors
        result: dict[str, torch.Tensor | float | None] = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.squeeze(0)

        # Trainer convention: 0=yes/relevant, 1=no/irrelevant
        result["label"] = 0 if sample.label == 1 else 1
        result["teacher_score"] = sample.teacher_score
        return result

    def _load_image(self, path_or_url: str):
        """Load PIL image from local path or URL."""
        import requests as http_requests
        from PIL import Image

        try:
            if path_or_url.startswith(("http://", "https://")):
                resp = http_requests.get(path_or_url, stream=True, timeout=30)
                resp.raise_for_status()
                return Image.open(resp.raw).convert("RGB")

            return Image.open(path_or_url).convert("RGB")
        except Exception as exc:
            logger.warning("Failed to load image {}: {}", path_or_url, exc)
            return None

    def _load_audio(self, path_or_url: str):
        """Load audio waveform (float32)."""
        import requests as http_requests
        import soundfile as sf

        try:
            if path_or_url.startswith(("http://", "https://")):
                resp = http_requests.get(path_or_url, timeout=30)
                resp.raise_for_status()
                data, sr = sf.read(io.BytesIO(resp.content), dtype="float32")
            else:
                data, sr = sf.read(path_or_url, dtype="float32")

            duration = len(data) / sr if data.ndim == 1 else data.shape[0] / sr
            if duration > self.config.max_audio_duration_s:
                max_samples = int(self.config.max_audio_duration_s * sr)
                data = data[:max_samples] if data.ndim == 1 else data[:max_samples, :]

            return data
        except Exception as exc:
            logger.warning("Failed to load audio {}: {}", path_or_url, exc)
            return None

    def _load_video_frames(self, path_or_url: str):
        """Load capped number of uniformly sampled video frames."""
        from .video import extract_frames, extract_frames_from_url

        try:
            if path_or_url.startswith(("http://", "https://")):
                return extract_frames_from_url(
                    path_or_url,
                    max_frames=self.config.max_video_frames_train,
                    max_duration_s=self.config.max_video_duration_s,
                )

            return extract_frames(
                path_or_url,
                max_frames=self.config.max_video_frames_train,
                max_duration_s=self.config.max_video_duration_s,
            )
        except Exception as exc:
            logger.warning("Failed to load video {}: {}", path_or_url, exc)
            return []


def collate_fn(batch: list[dict]) -> dict:
    """Batch collation with token padding and optional multimodal tensors.

    Notes:
    - Supports mixed sequence lengths.
    - For optional multimodal tensors (eg. pixel_values), all items in the batch
      must contain the key and have stack-compatible shapes.
    """
    if not batch:
        raise ValueError("collate_fn received an empty batch")

    max_len = max(sample["input_ids"].size(0) for sample in batch)
    padded_input_ids = []
    padded_attention_masks = []
    labels = []
    teacher_scores = []
    has_teacher_scores = False

    for sample in batch:
        ids = sample["input_ids"]
        mask = sample["attention_mask"]
        pad_len = max_len - ids.size(0)

        if pad_len > 0:
            ids = torch.cat([ids, torch.zeros(pad_len, dtype=ids.dtype)])
            mask = torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])

        padded_input_ids.append(ids)
        padded_attention_masks.append(mask)
        labels.append(sample["label"])

        if sample.get("teacher_score") is not None:
            has_teacher_scores = True
            teacher_scores.append(float(sample["teacher_score"]))
        else:
            teacher_scores.append(0.0)

    result: dict[str, torch.Tensor] = {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_masks),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

    if has_teacher_scores:
        result["teacher_scores"] = torch.tensor(teacher_scores, dtype=torch.float32)

    # Stack all additional tensor keys emitted by processor (pixel/audio grids, etc.)
    excluded = {"input_ids", "attention_mask", "label", "teacher_score"}
    extra_tensor_keys = sorted(
        {
            key
            for sample in batch
            for key, value in sample.items()
            if key not in excluded and isinstance(value, torch.Tensor)
        }
    )

    for key in extra_tensor_keys:
        present = [key in sample for sample in batch]
        if not all(present):
            if len(batch) == 1 and present[0]:
                result[key] = batch[0][key].unsqueeze(0)
                continue
            missing_idx = [i for i, ok in enumerate(present) if not ok]
            raise ValueError(
                f"Cannot collate mixed-modality batch for key '{key}'. "
                f"Missing in sample indices: {missing_idx}. "
                "Use per_device_train_batch_size=1 for mixed multimodal batches."
            )

        try:
            result[key] = torch.stack([sample[key] for sample in batch])
        except RuntimeError as exc:
            raise ValueError(
                f"Failed to stack tensor key '{key}'. "
                "Tensor shapes differ across samples; use batch_size=1 for this stage."
            ) from exc

    return result
