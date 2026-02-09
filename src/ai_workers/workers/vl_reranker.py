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

    def _score_pair(self, query: str, document: str) -> float:
        """Score a single query-document pair."""
        import torch

        messages = [
            {"role": "system", "content": RERANKER_PREFIX},
            {
                "role": "user",
                "content": f"<query>{query}</query>\n<document>{document}</document>",
            },
        ]

        text = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(
            self.model.device
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]
            yes_logit = logits[:, self.yes_token_id]
            no_logit = logits[:, self.no_token_id]
            score = torch.sigmoid(yes_logit - no_logit)

        return score.item()

    @modal.asgi_app()
    def serve(self):
        from fastapi import FastAPI
        from pydantic import BaseModel

        from ai_workers.common.auth import auth_middleware

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

        app.middleware("http")(auth_middleware)

        @app.get("/health")
        async def health():
            return {"status": "ok", "model": MODEL_LIGHT}

        @app.post("/v1/rerank", response_model=RerankResponse)
        async def rerank(request: RerankRequest):
            scored = []
            for i, doc in enumerate(request.documents):
                score = self._score_pair(request.query, doc)
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

    def _score_pair(self, query: str, document: str) -> float:
        import torch

        messages = [
            {"role": "system", "content": RERANKER_PREFIX},
            {
                "role": "user",
                "content": f"<query>{query}</query>\n<document>{document}</document>",
            },
        ]

        text = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(
            self.model.device
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]
            yes_logit = logits[:, self.yes_token_id]
            no_logit = logits[:, self.no_token_id]
            score = torch.sigmoid(yes_logit - no_logit)

        return score.item()

    @modal.asgi_app()
    def serve(self):
        from fastapi import FastAPI
        from pydantic import BaseModel

        from ai_workers.common.auth import auth_middleware

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

        app.middleware("http")(auth_middleware)

        @app.get("/health")
        async def health():
            return {"status": "ok", "model": MODEL_HEAVY}

        @app.post("/v1/rerank", response_model=RerankResponse)
        async def rerank(request: RerankRequest):
            scored = []
            for i, doc in enumerate(request.documents):
                score = self._score_pair(request.query, doc)
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
