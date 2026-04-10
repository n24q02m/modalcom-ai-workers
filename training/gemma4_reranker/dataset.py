"""PyTorch Dataset and collate function for multimodal reranker training.

MmRerankDataset loads JSONL samples and produces tokenized inputs
ready for the training loop. Handles text, image, audio, and video
modalities via AutoProcessor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import Dataset

from .data_pipeline import read_jsonl

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
        jsonl_path: Path,
        processor: Any,
        config: DataConfig,
    ):
        self.samples = read_jsonl(jsonl_path)
        self.processor = processor
        self.config = config
        self.sampling_rate = (
            processor.feature_extractor.sampling_rate
            if hasattr(processor, "feature_extractor")
            else 16000
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, img_path_or_url: str) -> Any:
        from PIL import Image

        if img_path_or_url.startswith("http"):
            from io import BytesIO

            import requests as http_requests

            response = http_requests.get(img_path_or_url, timeout=10)
            return Image.open(BytesIO(response.content)).convert("RGB")
        return Image.open(img_path_or_url).convert("RGB")

    def _load_audio(self, audio_path_or_url: str) -> tuple[Any, int]:
        import io

        import soundfile as sf

        if audio_path_or_url.startswith("http"):
            import requests as http_requests

            response = http_requests.get(audio_path_or_url, timeout=10)
            audio_bytes = io.BytesIO(response.content)
            data, sr = sf.read(audio_bytes)
        else:
            data, sr = sf.read(audio_path_or_url)
        return data, sr

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # Build multimodal prompt
        content_parts = []
        images = []
        audios = []
        videos = []

        # 1. Query
        content_parts.append({"type": "text", "text": f"<Query>\n{sample.query}\n</Query>\n"})
        if sample.query_image:
            content_parts.append({"type": "image", "image": sample.query_image})
            images.append(self._load_image(sample.query_image))
        if sample.query_audio:
            content_parts.append({"type": "audio", "audio": sample.query_audio})
            audio_data, _sr = self._load_audio(sample.query_audio)
            audios.append(audio_data)

        # 2. Document
        content_parts.append(
            {"type": "text", "text": f"<Document>\n{sample.positive}\n</Document>\n"}
        )
        if sample.positive_image:
            content_parts.append({"type": "image", "image": sample.positive_image})
            images.append(self._load_image(sample.positive_image))
        if sample.positive_audio:
            content_parts.append({"type": "audio", "audio": sample.positive_audio})
            audio_data, _sr = self._load_audio(sample.positive_audio)
            audios.append(audio_data)

        # 3. Instruction
        content_parts.append(
            {
                "type": "text",
                "text": 'Judge whether the Document is relevant to the Query. Answer only "yes" or "no".',
            }
        )

        # Process via AutoProcessor
        inputs = self.processor(
            text=[content_parts],
            images=images if images else None,
            audios=audios if audios else None,
            videos=videos if videos else None,
            padding=True,
            return_tensors="pt",
        )

        # Remove batch dim
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Label: 0 for yes (positive)
        inputs["labels"] = torch.tensor(0, dtype=torch.long)

        if sample.teacher_pos_score is not None:
            inputs["teacher_scores"] = torch.tensor(sample.teacher_pos_score, dtype=torch.float32)

        return inputs


def collate_fn(batch: list[dict]) -> dict:
    """Collate samples into a batch."""
    from torch.nn.utils.rnn import pad_sequence

    elem = batch[0]
    collated = {}
    for key in elem:
        if key == "labels" or key == "teacher_scores":
            collated[key] = torch.stack([d[key] for d in batch])
        elif isinstance(elem[key], torch.Tensor):
            # Pad sequences if needed (though processor should handle it)
            if elem[key].ndim > 0:
                collated[key] = pad_sequence([d[key] for d in batch], batch_first=True)
            else:
                collated[key] = torch.stack([d[key] for d in batch])
        else:
            collated[key] = [d[key] for d in batch]

    return collated
