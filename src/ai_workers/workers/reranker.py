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

    def _score_pair(self, model_name: str, query: str, document: str) -> float:
        """Score a single query-document pair using yes/no logit comparison."""
        import torch

        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]

        # Build chat messages following Qwen3-Reranker format
        messages = [
            {"role": "system", "content": RERANKER_PREFIX},
            {
                "role": "user",
                "content": f"<Query>\n{query}\n</Query>\n\n<Document>\n{document}\n</Document>",
            },
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits

            # Get logits for "yes" and "no" tokens
            yes_id = tokenizer.convert_tokens_to_ids("yes")
            no_id = tokenizer.convert_tokens_to_ids("no")
            yes_logit = logits[yes_id].float()
            no_logit = logits[no_id].float()

            # Sigmoid of (yes - no) gives relevance probability
            score = torch.sigmoid(yes_logit - no_logit).item()

        return score

    @modal.asgi_app()
    def serve(self):
        from fastapi import Body, FastAPI
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

        from ai_workers.common.auth import auth_middleware

        app.middleware("http")(auth_middleware)

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

            # Score each document against the query
            results = []
            for i, doc in enumerate(body.documents):
                score = self._score_pair(body.model, body.query, doc)
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
