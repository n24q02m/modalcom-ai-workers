"""Text Reranker workers using Custom FastAPI.

Qwen3-Reranker uses CausalLM with yes/no token logit scoring.
Exposes Cohere-compatible /v1/rerank endpoint.
Two apps: reranker_light (0.6B, T4) and reranker_heavy (8B, A10G).

LiteLLM integration:
  model: cohere/qwen3-reranker-0.6b
  api_base: https://<modal-url>
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

import modal
from fastapi import FastAPI, Request
from pydantic import BaseModel

from ai_workers.common.auth import verify_api_key
from ai_workers.common.images import MODELS_MOUNT_PATH, transformers_image
from ai_workers.common.r2 import get_modal_cloud_bucket_mount

# ---------------------------------------------------------------------------
# Shared constants & Models
# ---------------------------------------------------------------------------

SCALEDOWN_WINDOW = 300
KEEP_WARM = 0
MODEL_LIGHT = "qwen3-reranker-0.6b"
MODEL_HEAVY = "qwen3-reranker-8b"

r2_mount = get_modal_cloud_bucket_mount()

# Qwen3-Reranker chat template for relevance scoring
RERANKER_PREFIX = (
    "Given a query and a document, judge whether the document is relevant to the query. "
    "Answer only 'yes' or 'no'."
)


class DocumentResult(BaseModel):
    index: int
    relevance_score: float
    document: dict[str, str]


class RerankResponse(BaseModel):
    results: list[DocumentResult]
    model: str


class BaseRerankRequest(BaseModel):
    query: str
    documents: list[str]
    top_n: int | None = None


class RerankRequestLight(BaseRerankRequest):
    model: str = MODEL_LIGHT


class RerankRequestHeavy(BaseRerankRequest):
    model: str = MODEL_HEAVY


def create_reranker_app(
    title: str,
    model_name: str,
    request_model: type[BaseRerankRequest],
    score_fn: Callable[[str, str], float],
) -> FastAPI:
    app = FastAPI(title=title)

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        if request.url.path in ("/health", "/"):
            return await call_next(request)
        await verify_api_key(request)
        return await call_next(request)

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": model_name}

    @app.post("/v1/rerank", response_model=RerankResponse)
    async def rerank(request: request_model):
        results = []
        for i, doc in enumerate(request.documents):
            score = score_fn(request.query, doc)
            results.append(
                DocumentResult(
                    index=i,
                    relevance_score=score,
                    document={"text": doc},
                )
            )

        # Sort by score descending
        results.sort(key=lambda r: r.relevance_score, reverse=True)

        # Apply top_n
        if request.top_n is not None:
            results = results[: request.top_n]

        return RerankResponse(results=results, model=getattr(request, "model", model_name))

    return app


# ---------------------------------------------------------------------------
# Reranker Light (Qwen3-Reranker-0.6B, T4)
# ---------------------------------------------------------------------------

reranker_light_app = modal.App(
    "ai-workers-qwen3-reranker-0.6b",
    secrets=[modal.Secret.from_name("r2-credentials"), modal.Secret.from_name("worker-api-key")],
)


@reranker_light_app.cls(
    gpu="T4",
    image=transformers_image(),
    volumes={MODELS_MOUNT_PATH: r2_mount},
    scaledown_window=SCALEDOWN_WINDOW,
    keep_warm=KEEP_WARM,
    timeout=600,
    allow_concurrent_inputs=10,
)
class RerankerLightServer:
    """Custom FastAPI reranker server for Qwen3-Reranker-0.6B."""

    @modal.enter()
    def load_model(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = f"{MODELS_MOUNT_PATH}/{MODEL_LIGHT}"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )
        self.model.eval()

        # Pre-compute token IDs for "yes" and "no"
        self.yes_token_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.no_token_id = self.tokenizer.convert_tokens_to_ids("no")

    def _score_pair(self, query: str, document: str) -> float:
        """Score a single query-document pair using yes/no logits."""
        import torch

        messages = [
            {"role": "system", "content": RERANKER_PREFIX},
            {
                "role": "user",
                "content": f"Query: {query}\nDocument: {document}",
            },
        ]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]  # Last token logits
            yes_logit = logits[0, self.yes_token_id].float()
            no_logit = logits[0, self.no_token_id].float()
            # Sigmoid of (yes - no) gives relevance score in [0, 1]
            score = torch.sigmoid(yes_logit - no_logit).item()

        return score

    @modal.asgi_app()
    def serve(self):
        return create_reranker_app(
            title="Qwen3 Reranker Light",
            model_name=MODEL_LIGHT,
            request_model=RerankRequestLight,
            score_fn=self._score_pair,
        )


# ---------------------------------------------------------------------------
# Reranker Heavy (Qwen3-Reranker-8B, A10G)
# ---------------------------------------------------------------------------

reranker_heavy_app = modal.App(
    "ai-workers-qwen3-reranker-8b",
    secrets=[modal.Secret.from_name("r2-credentials"), modal.Secret.from_name("worker-api-key")],
)


@reranker_heavy_app.cls(
    gpu="A10G",
    image=transformers_image(),
    volumes={MODELS_MOUNT_PATH: r2_mount},
    scaledown_window=SCALEDOWN_WINDOW,
    keep_warm=KEEP_WARM,
    timeout=600,
    allow_concurrent_inputs=10,
)
class RerankerHeavyServer:
    """Custom FastAPI reranker server for Qwen3-Reranker-8B."""

    @modal.enter()
    def load_model(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = f"{MODELS_MOUNT_PATH}/{MODEL_HEAVY}"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )
        self.model.eval()
        self.yes_token_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.no_token_id = self.tokenizer.convert_tokens_to_ids("no")

    def _score_pair(self, query: str, document: str) -> float:
        import torch

        messages = [
            {"role": "system", "content": RERANKER_PREFIX},
            {"role": "user", "content": f"Query: {query}\nDocument: {document}"},
        ]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]
            yes_logit = logits[0, self.yes_token_id].float()
            no_logit = logits[0, self.no_token_id].float()
            score = torch.sigmoid(yes_logit - no_logit).item()

        return score

    @modal.asgi_app()
    def serve(self):
        return create_reranker_app(
            title="Qwen3 Reranker Heavy",
            model_name=MODEL_HEAVY,
            request_model=RerankRequestHeavy,
            score_fn=self._score_pair,
        )
