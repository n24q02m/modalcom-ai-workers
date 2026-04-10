"""Mining hard negatives for reranker training.

Uses a bi-encoder (embedding model) to find top-K documents for a query,
skips the absolute top (likely positives) and samples hard negatives
from the remaining top pool.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MiningTask:
    """Grouped parameters for mining a single query."""

    query: str
    positive: str
    corpus: list[str]
    modality: str
    query_image: str | None = None
    positive_image: str | None = None


class HardNegativeMiner:
    """Mines hard negatives using a teacher embedding model."""

    def __init__(self, model_id: str = "BAAI/bge-m3"):
        self.model_id = model_id
        self._load_model()

    def _load_model(self) -> None:
        """Lazy load embedding model."""
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.model_id)

    def embed_content(self, text: str, _modality: str) -> np.ndarray:
        """Embed text content."""
        return self.model.encode(text, convert_to_numpy=True)

    def get_teacher_score(self, query: str, doc: str, _modality: str) -> float:
        """Get cosine similarity from teacher model."""
        q_emb = self.embed_content(query, _modality)
        d_emb = self.embed_content(doc, _modality)
        return float(np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb)))

    def mine_negatives(self, task: MiningTask) -> list[str]:
        """Mine 7 hard negatives for a query."""
        query = task.query
        corpus = task.corpus
        modality = task.modality

        # 1. Embed query
        q_emb = self.embed_content(query, modality)

        # 2. Rank corpus (dot product similarity)
        if len(corpus) == 0:
            return []

        # Optimization: batch encode corpus if large
        if len(corpus) > 32:
            corpus_embeddings = self.model.encode(corpus, convert_to_numpy=True)
        else:
            corpus_embeddings = np.array([self.embed_content(doc, modality) for doc in corpus])

        scores = np.dot(corpus_embeddings, q_emb)
        ranked_indices = np.argsort(scores)[::-1]

        # 3. Hard Negative Selection (Skip top 10, pick 7 from 10-50)
        # Handle cases where corpus is small
        start_idx = min(10, len(ranked_indices))
        end_idx = min(50, len(ranked_indices))

        pool_indices = ranked_indices if start_idx == end_idx else ranked_indices[start_idx:end_idx]

        # Randomly select 7 hard negatives from the pool
        num_negatives = min(7, len(pool_indices))
        selected_neg_indices = np.random.choice(pool_indices, size=num_negatives, replace=False)
        return [corpus[i] for i in selected_neg_indices]
