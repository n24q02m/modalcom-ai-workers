"""Evaluation module for BEIR, MMEB, Audio, and Video benchmarks.

Provides reranking evaluation metrics:
- NDCG@10 for BEIR (text retrieval)
- Precision@1 for MMEB (multimodal)
- Recall@K for audio and video retrieval
- Regression checks between training stages
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalResult:
    """Result of a single evaluation benchmark."""

    benchmark: str
    metric_name: str
    metric_value: float
    dataset: str = ""
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize for JSONL logging."""
        return {
            "benchmark": self.benchmark,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "dataset": self.dataset,
            "details": self.details,
        }


def ndcg_at_k(relevances: list[float], k: int = 10) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank k.

    Args:
        relevances: List of relevance scores in ranked order.
        k: Cutoff rank.

    Returns:
        NDCG@k score (0.0 to 1.0).
    """
    relevances = relevances[:k]

    # DCG
    dcg = sum(
        rel / math.log2(i + 2)  # i+2 because log2(1)=0
        for i, rel in enumerate(relevances)
    )

    # Ideal DCG (sort by relevance descending)
    ideal = sorted(relevances, reverse=True)
    idcg = sum(
        rel / math.log2(i + 2)
        for i, rel in enumerate(ideal)
    )

    if idcg == 0:
        return 0.0
    return dcg / idcg


def precision_at_k(relevant_indices: list[int], retrieved_indices: list[int], k: int = 1) -> float:
    """Compute Precision@k.

    Args:
        relevant_indices: Set of ground-truth relevant indices.
        retrieved_indices: Ranked list of retrieved indices.
        k: Cutoff rank.

    Returns:
        Precision@k score (0.0 to 1.0).
    """
    retrieved_top_k = set(retrieved_indices[:k])
    relevant_set = set(relevant_indices)
    return len(retrieved_top_k & relevant_set) / k


def recall_at_k(relevant_indices: list[int], retrieved_indices: list[int], k: int = 5) -> float:
    """Compute Recall@k.

    Args:
        relevant_indices: Set of ground-truth relevant indices.
        retrieved_indices: Ranked list of retrieved indices.
        k: Cutoff rank.

    Returns:
        Recall@k score (0.0 to 1.0).
    """
    if not relevant_indices:
        return 0.0
    retrieved_top_k = set(retrieved_indices[:k])
    relevant_set = set(relevant_indices)
    return len(retrieved_top_k & relevant_set) / len(relevant_set)


def eval_beir_reranking(
    rerank_fn,
    queries: list[str],
    candidate_docs: list[list[str]],
    relevance_labels: list[list[int]],
    dataset_name: str = "beir",
) -> EvalResult:
    """Evaluate reranker on BEIR-style text retrieval.

    Pipeline: Pre-retrieved candidates -> Reranker -> NDCG@10

    Args:
        rerank_fn: Callable(query, docs) -> list[float] scores.
        queries: List of query strings.
        candidate_docs: List of candidate document lists per query.
        relevance_labels: Ground truth relevance per query-doc pair.
        dataset_name: Name of the dataset for logging.

    Returns:
        EvalResult with average NDCG@10.
    """
    ndcg_scores = []

    for query, docs, labels in zip(queries, candidate_docs, relevance_labels):
        scores = rerank_fn(query, docs)

        # Sort by score descending, get relevance in new order
        ranked = sorted(
            zip(scores, labels), key=lambda x: x[0], reverse=True
        )
        ranked_relevances = [r[1] for r in ranked]

        ndcg = ndcg_at_k(ranked_relevances, k=10)
        ndcg_scores.append(ndcg)

    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    return EvalResult(
        benchmark="BEIR",
        metric_name="NDCG@10",
        metric_value=avg_ndcg,
        dataset=dataset_name,
        details={"num_queries": len(queries), "scores": ndcg_scores},
    )


def eval_mmeb_reranking(
    rerank_fn,
    queries: list[str],
    candidate_docs: list[list[str]],
    relevant_indices: list[list[int]],
    dataset_name: str = "mmeb",
) -> EvalResult:
    """Evaluate reranker on MMEB-style multimodal retrieval.

    Args:
        rerank_fn: Callable(query, docs) -> list[float] scores.
        queries: Query strings.
        candidate_docs: Candidate lists per query.
        relevant_indices: Ground truth relevant doc indices.
        dataset_name: Dataset name for logging.

    Returns:
        EvalResult with average P@1.
    """
    p1_scores = []

    for query, docs, rel_idx in zip(queries, candidate_docs, relevant_indices):
        scores = rerank_fn(query, docs)

        # Rank by score descending
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )

        p1 = precision_at_k(rel_idx, ranked_indices, k=1)
        p1_scores.append(p1)

    avg_p1 = sum(p1_scores) / len(p1_scores) if p1_scores else 0.0

    return EvalResult(
        benchmark="MMEB",
        metric_name="P@1",
        metric_value=avg_p1,
        dataset=dataset_name,
        details={"num_queries": len(queries), "scores": p1_scores},
    )


def check_regression(
    current_results: list[EvalResult],
    baseline_results: list[EvalResult],
    max_regression: float = 0.05,
) -> list[str]:
    """Check for regression between training stages.

    Args:
        current_results: Eval results from current stage.
        baseline_results: Eval results from baseline (Stage 1).
        max_regression: Maximum allowed relative regression (e.g., 0.05 = 5%).

    Returns:
        List of warning strings for regressions exceeding threshold.
        Empty list means all checks passed.
    """
    warnings = []

    baseline_map = {
        (r.benchmark, r.dataset): r.metric_value for r in baseline_results
    }

    for result in current_results:
        key = (result.benchmark, result.dataset)
        if key not in baseline_map:
            continue

        baseline_val = baseline_map[key]
        if baseline_val == 0:
            continue

        regression = (baseline_val - result.metric_value) / baseline_val
        if regression > max_regression:
            warnings.append(
                f"REGRESSION: {result.benchmark}/{result.dataset} "
                f"{result.metric_name} dropped {regression:.1%} "
                f"({baseline_val:.4f} -> {result.metric_value:.4f}), "
                f"threshold: {max_regression:.1%}"
            )

    return warnings


def save_eval_results(results: list[EvalResult], path: Path) -> None:
    """Save evaluation results to JSONL file.

    Args:
        results: List of EvalResult objects.
        path: Output JSONL path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for result in results:
            json.dump(result.to_dict(), f, ensure_ascii=False)
            f.write("\n")


def load_eval_results(path: Path) -> list[EvalResult]:
    """Load evaluation results from JSONL file.

    Args:
        path: Input JSONL path.

    Returns:
        List of EvalResult objects.
    """
    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            results.append(EvalResult(**d))
    return results
