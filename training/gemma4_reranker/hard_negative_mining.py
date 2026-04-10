"""Hard negative mining + teacher scoring for grouped reranker dataset creation."""

from __future__ import annotations

import json
import logging
import os

import numpy as np
from google import genai
from google.genai import types
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from .data_pipeline import TrainSample
except ImportError:  # pragma: no cover - notebook/script fallback
    from data_pipeline import TrainSample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HardNegativeMining")

# =====================================================================
# Configuration
# =====================================================================
EMBEDDING_MODEL = "models/text-embedding-004"
MULTIMODAL_EMBEDDING_MODEL = "models/gemini-embedding-2-preview"
TEACHER_MODEL = "models/gemini-3-flash-preview"

class RelevanceOutput(BaseModel):
    relevance_score: float


class GeminiMiner:
    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("Must provide an API key (GOOGLE_API_KEY or GEMINI_API_KEY) in env vars.")

        self.client = genai.Client(api_key=key)

    @retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(5))
    def embed_content(self, content: str, modality: str = "text") -> np.ndarray:
        """Get embedding vector for content."""
        model_to_use = MULTIMODAL_EMBEDDING_MODEL if modality != "text" else EMBEDDING_MODEL

        response = self.client.models.embed_content(
            model=model_to_use,
            contents=content,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        return np.array(response.embeddings[0].values)

    @retry(wait=wait_exponential(multiplier=1, min=2, max=15), stop=stop_after_attempt(3))
    def get_teacher_score(self, query: str, document: str, modality: str = "text") -> float:
        """Query teacher model for relevance score in [0, 1]."""
        prompt = f"""You are an expert search and relevance ranking system.
Evaluate the relevance of the following Document to the provided Query.
Output EXACTLY a JSON payload with a single key 'relevance_score' containing a float between 0.0 (completely irrelevant) and 1.0 (highly relevant).

Query: {query}
Document: {document}
"""

        response = self.client.models.generate_content(
            model=TEACHER_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RelevanceOutput,
                temperature=0.1,
            ),
        )
        try:
            result = json.loads(response.text)
            return float(result.get("relevance_score", 0.0))
        except Exception as e:
            logger.warning(f"Failed to parse teacher score: {response.text}, returning 0.0. Error: {e}")
            return 0.0

    def process_query_sample(
        self,
        query: str,
        positive: str,
        corpus: list[str],
        corpus_embeddings: np.ndarray | None = None,
        modality: str = "text",
        language: str = "und",
        source: str = "unknown",
    ) -> TrainSample:
        """
        Execute Spec v5 Pipeline:
        1. Embed query (Multimodal Embedding 2)
        2. Cosine similarity against corpus
        3. Skip top 10, pick 7 from 10-50
        4. Score Positive with Gemini 3 Flash
        5. Score 7 Negatives with Gemini 3 Flash
        6. Return TrainSample
        """
        q_emb = self.embed_content(query, modality)

        if corpus_embeddings is None:
            corpus_embeddings = np.array([self.embed_content(doc, modality) for doc in corpus])

        scores = np.dot(corpus_embeddings, q_emb)
        ranked_indices = np.argsort(scores)[::-1]

        start_idx = min(10, len(ranked_indices))
        end_idx = min(50, len(ranked_indices))

        pool_indices = ranked_indices if start_idx == end_idx else ranked_indices[start_idx:end_idx]

        num_negatives = min(7, len(pool_indices))
        selected_neg_indices = np.random.choice(pool_indices, size=num_negatives, replace=False)
        negatives = [corpus[i] for i in selected_neg_indices]

        teacher_pos_score = self.get_teacher_score(query, positive, modality)
        teacher_neg_scores = [self.get_teacher_score(query, neg, modality) for neg in negatives]

        return TrainSample(
            query=query,
            positive=positive,
            negatives=negatives,
            modality=modality,
            language=language,
            source=source,
            teacher_pos_score=teacher_pos_score,
            teacher_neg_scores=teacher_neg_scores,
        )

