"""Text Reranker worker using Custom FastAPI.

Serves Qwen3-Reranker-8B on a single A10G container.
Uses AutoModelForCausalLM with yes/no logit scoring for relevance.

Scoring follows the official Qwen3-Reranker approach:
- Backbone-only forward pass (skip full lm_head matmul)
- Pre-extracted yes/no weight vectors from lm_head
- log_softmax([no, yes]) then exp(yes_prob) for score

Uses Modal Volume (pre-downloaded weights) + GPU Memory Snapshot
for fast cold start (~5-10s instead of >10 minutes).

LiteLLM integration:
  model: openai/qwen3-reranker-8b
  api_base: https://<modal-url>
"""

import modal

from ai_workers.common.config import get_model
from ai_workers.common.images import transformers_image
from ai_workers.common.volumes import HF_CACHE_DIR, hf_cache_vol

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCALEDOWN_WINDOW = 300  # 5 minutes
KEEP_WARM = 0  # Scale to zero when idle

# Model served by this app — loaded from HuggingFace Hub
MODEL_CONFIGS = {
    "qwen3-reranker-8b": {"hf_id": "Qwen/Qwen3-Reranker-8B"},
}

# Official system prompt for Qwen3-Reranker
RERANKER_SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query and the Instruct "
    'provided. Note that the answer can only be "yes" or "no".'
)

# Default instruction when none provided by caller
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

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
    """Reranker server for Qwen3-Reranker-8B.

    Uses yes/no logit scoring with sigmoid for relevance probability.
    """

    @modal.enter(snap=True)
    def load_models(self) -> None:
        """Load reranker model at container startup (snapshotted by GPU Memory Snapshot)."""
        import torch
        from loguru import logger
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.models: dict[str, object] = {}
        self.tokenizers: dict[str, object] = {}
        self.yes_no_weights: dict[str, object] = {}

        for name, cfg in MODEL_CONFIGS.items():
            hf_id = cfg["hf_id"]
            registry_cfg = get_model(name)
            logger.info("Loading {} ...", hf_id)
            tokenizer = AutoTokenizer.from_pretrained(
                hf_id,
                trust_remote_code=registry_cfg.trust_remote_code,
                padding_side="left",
                cache_dir=HF_CACHE_DIR,
            )
            model = AutoModelForCausalLM.from_pretrained(
                hf_id,
                torch_dtype=torch.float16,
                trust_remote_code=registry_cfg.trust_remote_code,
                device_map="auto",
                cache_dir=HF_CACHE_DIR,
            )
            model.eval()

            # Pre-extract yes/no weight vectors from lm_head for optimized scoring.
            # Instead of computing all 151,669 logits, we only compute 2.
            no_id = tokenizer.convert_tokens_to_ids("no")
            yes_id = tokenizer.convert_tokens_to_ids("yes")
            yes_no_weight = model.lm_head.weight.data[[no_id, yes_id], :].clone()

            self.models[name] = model
            self.tokenizers[name] = tokenizer
            self.yes_no_weights[name] = yes_no_weight
            logger.info(
                "Loaded {} successfully (yes_no_weight shape: {})", name, yes_no_weight.shape
            )

    def _score_pair(
        self, model_name: str, query: str, document: str, instruction: str | None = None
    ) -> float:
        """Score a single query-document pair using optimized yes/no logit scoring.

        Uses backbone-only forward pass + pre-extracted yes/no weights from lm_head,
        avoiding the full (hidden_dim x vocab_size) matmul.
        Scoring: log_softmax([no, yes]) then take yes probability.
        """
        import torch
        from torch.nn import functional as fn

        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        yes_no_weight = self.yes_no_weights[model_name]

        instruction = instruction or DEFAULT_INSTRUCTION

        # Build chat messages following official Qwen3-Reranker format
        messages = [
            {"role": "system", "content": RERANKER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}",
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
            # Backbone-only forward pass — skip lm_head matmul
            outputs = model.model(**inputs)
            hidden = outputs.last_hidden_state[:, -1, :]  # (1, hidden_dim)

            # Compute only yes/no logits using pre-extracted weights
            logits_2 = fn.linear(hidden, yes_no_weight)  # (1, 2) = [no_logit, yes_logit]

            # Official scoring: softmax then take yes probability
            probs = fn.softmax(logits_2.float(), dim=-1)
            score = probs[0, 1].item()  # Index 1 = yes

        return score

    def _score_batch(
        self, model_name: str, query: str, documents: list[str], instruction: str | None = None
    ) -> list[float]:
        """Score a batch of query-document pairs using optimized chunked inference.

        Batched alternative to `_score_pair` to prevent sequential processing bottleneck.
        Processes in chunks (e.g., batch_size=32) with padding and truncation.
        Uses right padding and extracts the last token hidden state via attention_mask sum.
        """
        import torch
        from torch.nn import functional as fn

        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        yes_no_weight = self.yes_no_weights[model_name]

        instruction = instruction or DEFAULT_INSTRUCTION

        # Ensure right padding for correct last-token extraction
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        all_scores = []
        batch_size = 32

        for i in range(0, len(documents), batch_size):
            chunk = documents[i : i + batch_size]
            messages_chunk = [
                [
                    {"role": "system", "content": RERANKER_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}",
                    },
                ]
                for doc in chunk
            ]

            texts = [
                tokenizer.apply_chat_template(
                    m,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for m in messages_chunk
            ]

            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(
                model.device
            )

            with torch.no_grad():
                outputs = model.model(**inputs)

                # Find the last non-padding token indices
                last_token_indices = inputs["attention_mask"].sum(1) - 1
                curr_batch_size = len(chunk)

                # Extract hidden states for the last tokens: (batch_size, hidden_dim)
                hidden = outputs.last_hidden_state[
                    torch.arange(curr_batch_size, device=model.device), last_token_indices, :
                ]

                # Compute only yes/no logits using pre-extracted weights
                logits_2 = fn.linear(
                    hidden, yes_no_weight
                )  # (batch_size, 2) = [no_logit, yes_logit]

                # Official scoring: softmax then take yes probability
                probs = fn.softmax(logits_2.float(), dim=-1)
                scores = probs[:, 1].tolist()  # Index 1 = yes
                all_scores.extend(scores)

        return all_scores

    @modal.asgi_app()
    def serve(self):
        from fastapi import Body, FastAPI, Request
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel, Field

        app = FastAPI(title="Qwen3 Reranker (8B)")

        class RerankRequest(BaseModel):
            model: str = "qwen3-reranker-8b"
            query: str
            documents: list[str] = Field(max_length=256)
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

        def _do_rerank(body: RerankRequest) -> RerankResponse:
            if body.model not in MODEL_CONFIGS:
                raise ValueError(
                    f"Unknown model: {body.model}. Available: {list(MODEL_CONFIGS.keys())}"
                )

            # Score all documents against the query in batches
            scores = self._score_batch(body.model, body.query, body.documents)

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
        def rerank_v1(body: RerankRequest = Body(...)):
            try:
                return _do_rerank(body)
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Invalid rerank request for model: {body.model}"},
                )

        @app.post("/v2/rerank", response_model=RerankResponse)
        def rerank_v2(body: RerankRequest = Body(...)):
            try:
                return _do_rerank(body)
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Invalid rerank request for model: {body.model}"},
                )

        return app
