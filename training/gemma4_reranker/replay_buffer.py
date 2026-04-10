"""Experience replay buffer for multi-stage training.

Prevents catastrophic forgetting by mixing a fraction (replay_ratio=10%)
of samples from previous stages into the current training data.

Buffer maintains a balanced sample from each completed modality type.
"""

from __future__ import annotations

import random
from pathlib import Path

from .data_pipeline import TrainSample, read_jsonl


class ReplayBuffer:
    """Manages experience replay samples across training stages.

    Maintains a fixed-size buffer per modality to prevent catastrophic
    forgetting during incremental multi-stage training.
    """

    def __init__(self, max_per_modality: int = 20, seed: int = 42) -> None:
        """Initialize replay buffer.

        Args:
            max_per_modality: Maximum samples to keep per modality type.
            seed: Random seed for reproducible sampling.
        """
        self._buffer: dict[str, list[TrainSample]] = {}
        self._max_per_modality = max_per_modality
        self._rng = random.Random(seed)

    @property
    def total_size(self) -> int:
        """Total number of samples across all modalities."""
        return sum(len(v) for v in self._buffer.values())

    @property
    def modalities(self) -> list[str]:
        """List of modality types in the buffer."""
        return list(self._buffer.keys())

    def ingest(self, samples: list[TrainSample]) -> int:
        """Add samples to the buffer, maintaining per-modality limits.

        If a modality exceeds max_per_modality, randomly downsample.

        Args:
            samples: Training samples to add.

        Returns:
            Total number of samples now in buffer.
        """
        for sample in samples:
            mod = sample.modality
            if mod not in self._buffer:
                self._buffer[mod] = []
            self._buffer[mod].append(sample)

        # Enforce limits per modality
        for mod in self._buffer:
            if len(self._buffer[mod]) > self._max_per_modality:
                self._buffer[mod] = self._rng.sample(self._buffer[mod], self._max_per_modality)

        return self.total_size

    def ingest_from_file(self, path: Path) -> int:
        """Load and ingest samples from a JSONL file.

        Args:
            path: Path to JSONL training data file.

        Returns:
            Total number of samples now in buffer.
        """
        samples = read_jsonl(path)
        return self.ingest(samples)

    def sample(self, n: int) -> list[TrainSample]:
        """Sample n items from the buffer, balanced across modalities.

        Args:
            n: Number of samples to draw.

        Returns:
            List of TrainSample drawn from the buffer.
            May return fewer than n if buffer is smaller.
        """
        all_samples = []
        for mod_samples in self._buffer.values():
            all_samples.extend(mod_samples)

        if not all_samples:
            return []

        n = min(n, len(all_samples))
        return self._rng.sample(all_samples, n)

    def get_replay_samples(self, train_size: int, replay_ratio: float) -> list[TrainSample]:
        """Get replay samples proportional to training set size.

        Args:
            train_size: Number of samples in the current stage training set.
            replay_ratio: Fraction of training data to replace with replay
                         (e.g., 0.1 = 10%).

        Returns:
            List of replay TrainSample to mix into training.
        """
        if replay_ratio <= 0 or self.total_size == 0:
            return []

        n_replay = max(1, int(train_size * replay_ratio))
        return self.sample(n_replay)

    def clear(self) -> None:
        """Clear all samples from the buffer."""
        self._buffer.clear()

    def stats(self) -> dict[str, int]:
        """Get per-modality sample counts.

        Returns:
            Dict mapping modality name to sample count.
        """
        return {mod: len(samples) for mod, samples in self._buffer.items()}
