"""
Hard Negative Mining & Teacher Scoring utilizing Google Native APIs (Kaggle/Google AI Studio).
- Dense Retrieval: Gemini Multimodal Embedding (Native support for Text, Image, Audio, Video)
- Teacher Scoring: Gemini 3 Flash (SOTA, replacing 1.5)
"""

import json
import logging
import os

import numpy as np
from data_pipeline import TrainSample
from pydantic import BaseModel

# Conditional import to handle missing dependencies in test environment
try:
    import google.generativeai as genai
    from tenacity import retry, stop_after_attempt, wait_exponential

    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

    # Dummy decorators for when tenacity is missing
    def retry(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def wait_exponential(*args, **kwargs):
        return None

    def stop_after_attempt(*args, **kwargs):
        return None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HardNegativeMining")

# =====================================================================
# CONFIGURATION: Gemini 3 Flash & Multimodal Embeddings
# =====================================================================
# For dense retrieval across modalities
EMBEDDING_MODEL = "models/text-embedding-004"  # Text fallback
MULTIMODAL_EMBEDDING_MODEL = "models/gemini-embedding-2-preview"  # Native MM Gemini Embedding 2

# For Teacher Reranking / Knowledge Distillation
TEACHER_MODEL = "models/gemini-3-flash-preview"  # Latest Gemini 3 Flash


class RelevanceOutput(BaseModel):
    relevance_score: float


class MiningTask(BaseModel):
    """Container for hard negative mining input parameters."""

    query: str
    positive: str
    corpus: list[str]
    modality: str = "text"


class GeminiMiner:
    def __init__(self, api_key: str | None = None):
        if not HAS_GENAI:
            logger.warning(
                "google-generativeai or tenacity not installed. GeminiMiner will not function correctly."
            )
            return

        key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "Must provide an API key (GOOGLE_API_KEY or GEMINI_API_KEY) in env vars."
            )
        genai.configure(api_key=key)

        self.teacher_client = genai.GenerativeModel(TEACHER_MODEL)

    @retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(5))
    def embed_content(self, content: str, modality: str = "text") -> np.ndarray:
        """Get embeddings using Gemini 2 Multimodal Embeddings."""
        if not HAS_GENAI:
            raise RuntimeError("google-generativeai not installed.")

        # Note: Depending on modality, you'd pass specific mime types and byte data.
        # For simplicity in this text mock, using text-embedding.
        model_to_use = MULTIMODAL_EMBEDDING_MODEL if modality != "text" else EMBEDDING_MODEL

        response = genai.embed_content(
            model=model_to_use, content=content, task_type="retrieval_document"
        )
        return np.array(response["embedding"])

    @retry(wait=wait_exponential(multiplier=1, min=2, max=15), stop=stop_after_attempt(3))
    def get_teacher_score(self, query: str, document: str, modality: str = "text") -> float:
        """
        Ask Gemini 3 Flash to act as a Teacher Reranker and output a float score [0.0, 1.0].
        Returns relevance float.
        """
        if not HAS_GENAI:
            raise RuntimeError("google-generativeai not installed.")

        prompt = f"""You are an expert search and relevance ranking system.
Evaluate the relevance of the following Document to the provided Query.
Output EXACTLY a JSON payload with a single key 'relevance_score' containing a float between 0.0 (completely irrelevant) and 1.0 (highly relevant).

Query: {query}
Document: {document}
"""
        # Feature of Gemini 3 / 1.5+: Structured structured outputs using response_schema
        response = self.teacher_client.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=RelevanceOutput,
                temperature=0.1,
            ),
        )
        try:
            result = json.loads(response.text)
            return float(result.get("relevance_score", 0.0))
        except Exception as e:
            logger.warning(
                f"Failed to parse teacher score: {response.text}, returning 0.0. Error: {e}"
            )
            return 0.0

    def process_query_sample(
        self,
        task: MiningTask,
        corpus_embeddings: np.ndarray | None = None,
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
        # 1. Embed Query
        q_emb = self.embed_content(task.query, task.modality)

        # 2. Dense Retrieval (Cosine Sim)
        if corpus_embeddings is None:
            # Fallback: embed the whole corpus right now (slow for large data, usually pre-computed)
            corpus_embeddings = np.array(
                [self.embed_content(doc, task.modality) for doc in task.corpus]
            )

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
        negatives = [task.corpus[i] for i in selected_neg_indices]

        # 4 & 5. Teacher Scoring via Gemini 3 Flash
        teacher_pos_score = self.get_teacher_score(task.query, task.positive, task.modality)
        teacher_neg_scores = [
            self.get_teacher_score(task.query, neg, task.modality) for neg in negatives
        ]

        # 6. Construct TrainSample from data_pipeline.py
        return TrainSample(
            query=task.query,
            positive=task.positive,
            negatives=negatives,
            modality=task.modality,
            teacher_pos_score=teacher_pos_score,
            teacher_neg_scores=teacher_neg_scores,
        )
