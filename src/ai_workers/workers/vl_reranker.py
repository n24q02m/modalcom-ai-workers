"""Vision-Language Reranker workers using Custom FastAPI.

Qwen3-VL-Reranker uses a chat-based yes/no scoring approach (like text reranker)
but with multimodal input support.
Exposes Cohere-compatible /v1/rerank endpoint.
Two apps: vl_reranker_light (2B, T4) and vl_reranker_heavy (8B, A10G).

LiteLLM integration:
  model: cohere/qwen3-vl-reranker-2b
  api_base: https://<modal-url>
"""

from __future__ import annotations

import modal

from ai_workers.common.images import MODELS_MOUNT_PATH, transformers_image
from ai_workers.common.r2 import get_modal_cloud_bucket_mount

SCALEDOWN_WINDOW = 300
KEEP_WARM = 0

RERANKER_PREFIX = (
    "Given a query and a document with text and/or image, "
    "judge whether the document is relevant to the query. "
    "Answer 'yes' or 'no'."
)

r2_mount = get_modal_cloud_bucket_mount()


# ---------------------------------------------------------------------------
# VL Reranker Light (Qwen3-VL-Reranker-2B, T4)
# ---------------------------------------------------------------------------

vl_reranker_light_app = modal.App(
    "ai-workers-qwen3-vl-reranker-2b",
    secrets=[modal.Secret.from_name("r2-credentials"), modal.Secret.from_name("worker-api-key")],
)

MODEL_LIGHT = "qwen3-vl-reranker-2b"


@vl_reranker_light_app.cls(
    gpu="T4",
    image=transformers_image(),
    volumes={MODELS_MOUNT_PATH: r2_mount},
    scaledown_window=SCALEDOWN_WINDOW,
    keep_warm=KEEP_WARM,
    timeout=600,
    allow_concurrent_inputs=10,
)
class VLRerankerLightServer:
    """Custom FastAPI VL reranker server for Qwen3-VL-Reranker-2B."""

    @modal.enter()
    def load_model(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        model_path = f"{MODELS_MOUNT_PATH}/{MODEL_LIGHT}"
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )
        self.model.eval()

        # Cache yes/no token IDs
        self.yes_token_id = self.processor.tokenizer.convert_tokens_to_ids("yes")
        self.no_token_id = self.processor.tokenizer.convert_tokens_to_ids("no")

    def _score_batch(self, query: str, documents: list[str], batch_size: int = 16) -> list[float]:
        """Score a batch of query-document pairs."""
        import torch

        all_scores = []

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]

            texts = []
            for doc in batch_docs:
                messages = [
                    {"role": "system", "content": RERANKER_PREFIX},
                    {
                        "role": "user",
                        "content": f"<query>{query}</query>\n<document>{doc}</document>",
                    },
                ]
                text = self.processor.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                texts.append(text)

            inputs = self.processor(
                text=texts, return_tensors="pt", padding=True, truncation=True
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Get last token logits
                last_token_indices = inputs.attention_mask.sum(1) - 1
                batch_indices = torch.arange(len(batch_docs), device=self.model.device)

                logits = outputs.logits[batch_indices, last_token_indices, :]
                yes_logit = logits[:, self.yes_token_id]
                no_logit = logits[:, self.no_token_id]
                scores = torch.sigmoid(yes_logit - no_logit)

                all_scores.extend(scores.tolist())

        return all_scores

    @modal.asgi_app()
    def serve(self):
        from fastapi import FastAPI, Request
        from pydantic import BaseModel

        app = FastAPI(title="Qwen3 VL Reranker Light")

        class RerankRequest(BaseModel):
            query: str
            documents: list[str]
            top_n: int | None = None
            model: str = MODEL_LIGHT
            return_documents: bool = True

        class RerankResult(BaseModel):
            index: int
            relevance_score: float
            document: str | None = None

        class RerankResponse(BaseModel):
            results: list[RerankResult]
            model: str

        @app.middleware("http")
        async def auth_middleware(request: Request, call_next):
            if request.url.path in ("/health", "/"):
                return await call_next(request)
            from ai_workers.common.auth import verify_api_key

            await verify_api_key(request)
            return await call_next(request)

        @app.get("/health")
        async def health():
            return {"status": "ok", "model": MODEL_LIGHT}

        @app.post("/v1/rerank", response_model=RerankResponse)
        async def rerank(request: RerankRequest):
            scores = self._score_batch(request.query, request.documents)

            scored = []
            for i, (score, doc) in enumerate(zip(scores, request.documents, strict=True)):
                scored.append((i, score, doc))

            scored.sort(key=lambda x: x[1], reverse=True)

            top_n = request.top_n or len(scored)
            results = [
                RerankResult(
                    index=idx,
                    relevance_score=round(score, 6),
                    document=doc if request.return_documents else None,
                )
                for idx, score, doc in scored[:top_n]
            ]

            return RerankResponse(results=results, model=request.model)

        return app


# ---------------------------------------------------------------------------
# VL Reranker Heavy (Qwen3-VL-Reranker-8B, A10G)
# ---------------------------------------------------------------------------

vl_reranker_heavy_app = modal.App(
    "ai-workers-qwen3-vl-reranker-8b",
    secrets=[modal.Secret.from_name("r2-credentials"), modal.Secret.from_name("worker-api-key")],
)

MODEL_HEAVY = "qwen3-vl-reranker-8b"


@vl_reranker_heavy_app.cls(
    gpu="A10G",
    image=transformers_image(),
    volumes={MODELS_MOUNT_PATH: r2_mount},
    scaledown_window=SCALEDOWN_WINDOW,
    keep_warm=KEEP_WARM,
    timeout=600,
    allow_concurrent_inputs=10,
)
class VLRerankerHeavyServer:
    """Custom FastAPI VL reranker server for Qwen3-VL-Reranker-8B."""

    @modal.enter()
    def load_model(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        model_path = f"{MODELS_MOUNT_PATH}/{MODEL_HEAVY}"
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )
        self.model.eval()

        self.yes_token_id = self.processor.tokenizer.convert_tokens_to_ids("yes")
        self.no_token_id = self.processor.tokenizer.convert_tokens_to_ids("no")

    def _score_batch(self, query: str, documents: list[str], batch_size: int = 16) -> list[float]:
        """Score a batch of query-document pairs."""
        import torch

        all_scores = []

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]

            texts = []
            for doc in batch_docs:
                messages = [
                    {"role": "system", "content": RERANKER_PREFIX},
                    {
                        "role": "user",
                        "content": f"<query>{query}</query>\n<document>{doc}</document>",
                    },
                ]
                text = self.processor.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                texts.append(text)

            inputs = self.processor(
                text=texts, return_tensors="pt", padding=True, truncation=True
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Get last token logits
                last_token_indices = inputs.attention_mask.sum(1) - 1
                batch_indices = torch.arange(len(batch_docs), device=self.model.device)

                logits = outputs.logits[batch_indices, last_token_indices, :]
                yes_logit = logits[:, self.yes_token_id]
                no_logit = logits[:, self.no_token_id]
                scores = torch.sigmoid(yes_logit - no_logit)

                all_scores.extend(scores.tolist())

        return all_scores

    @modal.asgi_app()
    def serve(self):
        from fastapi import FastAPI, Request
        from pydantic import BaseModel

        app = FastAPI(title="Qwen3 VL Reranker Heavy")

        class RerankRequest(BaseModel):
            query: str
            documents: list[str]
            top_n: int | None = None
            model: str = MODEL_HEAVY
            return_documents: bool = True

        class RerankResult(BaseModel):
            index: int
            relevance_score: float
            document: str | None = None

        class RerankResponse(BaseModel):
            results: list[RerankResult]
            model: str

        @app.middleware("http")
        async def auth_middleware(request: Request, call_next):
            if request.url.path in ("/health", "/"):
                return await call_next(request)
            from ai_workers.common.auth import verify_api_key

            await verify_api_key(request)
            return await call_next(request)

        @app.get("/health")
        async def health():
            return {"status": "ok", "model": MODEL_HEAVY}

        @app.post("/v1/rerank", response_model=RerankResponse)
        async def rerank(request: RerankRequest):
            scores = self._score_batch(request.query, request.documents)

            scored = []
            for i, (score, doc) in enumerate(zip(scores, request.documents, strict=True)):
                scored.append((i, score, doc))

            scored.sort(key=lambda x: x[1], reverse=True)

            top_n = request.top_n or len(scored)
            results = [
                RerankResult(
                    index=idx,
                    relevance_score=round(score, 6),
                    document=doc if request.return_documents else None,
                )
                for idx, score, doc in scored[:top_n]
            ]

            return RerankResponse(results=results, model=request.model)

        return app
