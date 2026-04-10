"""PyTorch Dataset and collate function for multimodal reranker training.

MmRerankDataset loads JSONL samples and produces tokenized inputs
ready for the training loop. Handles text, image, audio, and video
modalities via AutoProcessor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

from .data_pipeline import TrainSample, read_jsonl

if TYPE_CHECKING:
    from pathlib import Path

    from .config import DataConfig


class MmRerankDataset(Dataset):
    """Dataset for multimodal reranker training.

    Each sample produces tokenized inputs with labels for
    yes/no classification (0=yes (relevant), 1=no (irrelevant)).
    """

    def __init__(
        self,
        data_path: Path,
        processor,
        config: DataConfig,
        reranker_prefix: str,
    ) -> None:
        """Initialize dataset.

        Args:
            data_path: Path to JSONL training data.
            processor: AutoProcessor instance for tokenization.
            config: Data configuration with limits.
            reranker_prefix: System prompt for relevance judgment.
        """
        self.samples = read_jsonl(data_path)
        self.processor = processor
        self.config = config
        self.reranker_prefix = reranker_prefix

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """Get a single tokenized training sample.

        Returns dict with:
            - input_ids: (seq_len,) token IDs
            - attention_mask: (seq_len,) mask
            - label: int (0=yes/relevant, 1=no/irrelevant)
            - teacher_score: float or None
            - images: list[PIL.Image] or None
            - audios: list[tuple[ndarray, int]] or None
        """
        sample = self.samples[idx]
        return self._process_sample(sample)

    def _process_sample(self, sample: TrainSample) -> dict:
        """Process a single TrainSample into model inputs."""
        content_parts = []
        images = []
        audios = []

        # Query media
        if sample.query_image:
            content_parts.append({"type": "image", "image": sample.query_image})
            images.append(self._load_image(sample.query_image))
        if sample.query_audio:
            content_parts.append({"type": "audio", "audio": sample.query_audio})
            audio_data, sr = self._load_audio(sample.query_audio)
            audios.append(audio_data)
        if sample.query_video:
            frames = self._load_video_frames(sample.query_video)
            for frame in frames:
                content_parts.append({"type": "image", "image": frame})
                images.append(frame)

        content_parts.append(
            {"type": "text", "text": f"<Query>\n{sample.query}\n</Query>"}
        )

        # Document media
        if sample.doc_image:
            content_parts.append({"type": "image", "image": sample.doc_image})
            images.append(self._load_image(sample.doc_image))
        if sample.doc_audio:
            content_parts.append({"type": "audio", "audio": sample.doc_audio})
            audio_data, _sr = self._load_audio(sample.doc_audio)
            audios.append(audio_data)
        if sample.doc_video:
            frames = self._load_video_frames(sample.doc_video)
            for frame in frames:
                content_parts.append({"type": "image", "image": frame})
                images.append(frame)

        content_parts.append(
            {"type": "text", "text": f"\n<Document>\n{sample.document}\n</Document>"}
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

        # Build processor kwargs
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

        # Squeeze batch dimension (single sample)
        result = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label": 0 if sample.label == 1 else 1,  # 0=yes, 1=no
            "teacher_score": sample.teacher_score,
        }

        # Pass optional pixel_values if present
        if "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"].squeeze(0)

        return result

    def _load_image(self, path_or_url: str):
        """Load PIL Image from local path or URL."""
        from PIL import Image

        if path_or_url.startswith(("http://", "https://")):
            import requests as http_requests
            resp = http_requests.get(path_or_url, stream=True, timeout=30)
            resp.raise_for_status()
            return Image.open(resp.raw).convert("RGB")

        return Image.open(path_or_url).convert("RGB")

    def _load_audio(self, path_or_url: str) -> tuple:
        """Load audio as (ndarray, sample_rate)."""
        import io

        import soundfile as sf

        if path_or_url.startswith(("http://", "https://")):
            import requests as http_requests
            resp = http_requests.get(path_or_url, timeout=30)
            resp.raise_for_status()
            buf = io.BytesIO(resp.content)
            data, sr = sf.read(buf, dtype="float32")
        else:
            data, sr = sf.read(path_or_url, dtype="float32")

        # Enforce training duration limit
        duration = len(data) / sr if data.ndim == 1 else data.shape[0] / sr
        if duration > self.config.max_audio_duration_s:
            # Truncate to max duration
            max_samples = int(self.config.max_audio_duration_s * sr)
            data = data[:max_samples] if data.ndim == 1 else data[:max_samples, :]

        return data, sr

    def _load_video_frames(self, path_or_url: str):
        """Load video frames (training: max 4 frames)."""
        from .video import extract_frames, extract_frames_from_url

        max_frames = self.config.max_video_frames_train
        max_duration = self.config.max_video_duration_s

        if path_or_url.startswith(("http://", "https://")):
            return extract_frames_from_url(
                path_or_url,
                max_frames=max_frames,
                max_duration_s=max_duration,
            )

        return extract_frames(
            path_or_url,
            max_frames=max_frames,
            max_duration_s=max_duration,
        )


def collate_fn(batch: list[dict]) -> dict:
    """Collate function for DataLoader with padding.

    Pads input_ids and attention_mask to the longest sequence in the batch.
    Handles variable-length sequences from mixed modalities.

    Args:
        batch: List of sample dicts from MmRerankDataset.__getitem__.

    Returns:
        Dict with batched tensors:
            - input_ids: (B, max_len)
            - attention_mask: (B, max_len)
            - labels: (B,)
            - teacher_scores: (B,) or None
    """
    # Find max sequence length
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

        # Pad on the right with zeros
        if pad_len > 0:
            ids = torch.cat([ids, torch.zeros(pad_len, dtype=ids.dtype)])
            mask = torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])

        padded_input_ids.append(ids)
        padded_attention_masks.append(mask)
        labels.append(sample["label"])

        if sample["teacher_score"] is not None:
            has_teacher_scores = True
            teacher_scores.append(sample["teacher_score"])
        else:
            teacher_scores.append(0.0)

    result = {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_masks),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

    if has_teacher_scores:
        result["teacher_scores"] = torch.tensor(teacher_scores, dtype=torch.float32)

    return result
