"""Text Reranker worker using Custom FastAPI (merged light + heavy).

Serves both Qwen3-Reranker-0.6B (light) and Qwen3-Reranker-8B (heavy)
from a single A10G container. Routes by ``model`` field in request.

Both models loaded at startup (~17GB total on A10G 24GB VRAM).
Uses AutoModelForCausalLM with yes/no logit scoring for relevance.

Uses Modal Volume (pre-downloaded weights) + GPU Memory Snapshot
for fast cold start (~5-10s instead of >10 minutes).

LiteLLM integration:
  model: openai/qwen3-reranker-0.6b  (or qwen3-reranker-8b)
  api_base: https://<modal-url>
"""

import modal

from ai_workers.common.images import transformers_image
from ai_workers.common.volumes import HF_CACHE_DIR, hf_cache_vol

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCALEDOWN_WINDOW = 300  # 5 minutes
KEEP_WARM = 0  # Scale to zero when idle

# Models served by this single app — loaded from HuggingFace Hub
MODEL_CONFIGS = {
    "qwen3-reranker-0.6b": {"hf_id": "Qwen/Qwen3-Reranker-0.6B"},
    "qwen3-reranker-8b": {"hf_id": "Qwen/Qwen3-Reranker-8B"},
}

# System prompt required by Qwen3-Reranker
RERANKER_PREFIX = 'Judge whether the Document is relevant to the Query. Answer only "yes" or "no".'

reranker_app = modal.App(
    "ai-workers-reranker",
    secrets=[modal.Secret.from_name("worker-api-key")],
)


@reranker_app.cls(
    gpu="A10G",
    image=transformers_image(),
    volumes={HF_CACHE_DIR: hf_cache_vol},
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=KEEP_WARM,
    timeout=1800,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=100)
class RerankerServer:
    """Merged reranker server for Qwen3-Reranker-0.6B + 8B.

    Both models loaded at startup. Routes request to correct model
    via the ``model`` field. Uses yes/no logit scoring with sigmoid
    for relevance probability.
    """

    @modal.enter(snap=True)
    def load_models(self) -> None:
        """Load both reranker models at container startup (snapshotted by GPU Memory Snapshot)."""
        import torch
        from loguru import logger
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.models: dict[str, object] = {}
        self.tokenizers: dict[str, object] = {}

        for name, cfg in MODEL_CONFIGS.items():
            hf_id = cfg["hf_id"]
            logger.info("Loading {} ...", hf_id)
            tokenizer = AutoTokenizer.from_pretrained(
                hf_id,
                trust_remote_code=True,
                cache_dir=HF_CACHE_DIR,
            )
            model = AutoModelForCausalLM.from_pretrained(
                hf_id,
                dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto",
                cache_dir=HF_CACHE_DIR,
            )
            model.eval()
            self.models[name] = model
            self.tokenizers[name] = tokenizer
            logger.info("Loaded {} successfully", name)

    def _score_pairs(self, model_name: str, query: str, documents: list[str]) -> list[float]:
        """Score a list of query-document pairs in batches using yes/no logit comparison."""
        import torch

        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]

        tokenizer.padding_side = "right"

        texts = []
        for doc in documents:
            messages = [
                {"role": "system", "content": RERANKER_PREFIX},
                {
                    "role": "user",
                    "content": f"<Query>\n{query}\n</Query>\n\n<Document>\n{doc}\n</Document>",
                },
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            texts.append(text)

        batch_size = 32
        all_scores = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(
                model.device
            )

            with torch.no_grad():
                outputs = model(**inputs)

                # Extract logits for the last valid token of each sequence
                attention_mask = inputs["attention_mask"]
                last_token_indices = attention_mask.sum(1) - 1
                current_batch_size = len(batch_texts)

                logits = outputs.logits[
                    torch.arange(current_batch_size, device=model.device), last_token_indices, :
                ]

                # Get logits for "yes" and "no" tokens
                yes_id = tokenizer.convert_tokens_to_ids("yes")
                no_id = tokenizer.convert_tokens_to_ids("no")
                yes_logits = logits[:, yes_id].float()
                no_logits = logits[:, no_id].float()

                # Sigmoid of (yes - no) gives relevance probability
                scores = torch.sigmoid(yes_logits - no_logits).tolist()
                all_scores.extend(scores)

        return all_scores

    @modal.asgi_app()
    def serve(self):
        from fastapi import Body, FastAPI, Request
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel

        app = FastAPI(title="Qwen3 Reranker (Light + Heavy)")

        class RerankPair(BaseModel):
            document: str

        class RerankRequest(BaseModel):
            model: str = "qwen3-reranker-0.6b"
            query: str
            documents: list[str]
            top_n: int | None = None
            return_documents: bool = False
            rank_fields: list[str] | None = None
            max_tokens_per_doc: int | None = None

        class RerankResultDocument(BaseModel):
            text: str

        class RerankResult(BaseModel):
            index: int
            relevance_score: float
            document: RerankResultDocument | None = None

        class RerankResponse(BaseModel):
            model: str
            results: list[RerankResult]

        @app.middleware("http")
        async def auth_middleware(request: Request, call_next):
            if request.url.path in ("/health", "/"):
                return await call_next(request)
            from fastapi import HTTPException as _HTTPException

            from ai_workers.common.auth import verify_api_key

            try:
                await verify_api_key(request)
            except _HTTPException as exc:
                return JSONResponse(
                    status_code=exc.status_code,
                    content={"detail": exc.detail},
                )
            return await call_next(request)

        @app.get("/health")
        async def health():
            return {
                "status": "ok",
                "models": list(MODEL_CONFIGS.keys()),
            }

        async def _do_rerank(body: RerankRequest) -> RerankResponse:
            if body.model not in MODEL_CONFIGS:
                raise ValueError(
                    f"Unknown model: {body.model}. Available: {list(MODEL_CONFIGS.keys())}"
                )

            if not body.documents:
                return RerankResponse(model=body.model, results=[])

            # Score all documents against the query in batches
            scores = self._score_pairs(body.model, body.query, body.documents)

            results = []
            for i, (doc, score) in enumerate(zip(body.documents, scores, strict=True)):
                result = RerankResult(index=i, relevance_score=score)
                if body.return_documents:
                    result.document = RerankResultDocument(text=doc)
                results.append(result)

            # Sort by relevance score descending
            results.sort(key=lambda x: x.relevance_score, reverse=True)

            # Apply top_n if specified
            if body.top_n is not None:
                results = results[: body.top_n]

            return RerankResponse(model=body.model, results=results)

        @app.post("/v1/rerank", response_model=RerankResponse)
        async def rerank_v1(body: RerankRequest = Body(...)):
            try:
                return await _do_rerank(body)
            except ValueError as e:
                return JSONResponse(status_code=400, content={"error": str(e)})

        @app.post("/v2/rerank", response_model=RerankResponse)
        async def rerank_v2(body: RerankRequest = Body(...)):
            try:
                return await _do_rerank(body)
            except ValueError as e:
                return JSONResponse(status_code=400, content={"error": str(e)})

        return app
