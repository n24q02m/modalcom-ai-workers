"""Vision-Language Reranker worker using Custom FastAPI.

Serves Qwen3-VL-Reranker-8B on a single A10G container.
Uses AutoModelForImageTextToText with yes/no logit scoring for relevance.

Scoring follows the official Qwen3-VL-Reranker approach:
- Backbone-only forward pass (skip full lm_head matmul)
- Pre-extracted yes/no weight vectors from lm_head
- softmax([no, yes]) then take yes probability for score

Supports text-only and image+text multimodal query/document pairs.

Uses Modal Volume (pre-downloaded weights) + GPU Memory Snapshot
for fast cold start (~5-10s instead of >10 minutes).
"""

from typing import Any

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
    "qwen3-vl-reranker-8b": {"hf_id": "Qwen/Qwen3-VL-Reranker-8B"},
}

# Official system prompt for Qwen3-VL-Reranker
RERANKER_SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query and the Instruct "
    'provided. Note that the answer can only be "yes" or "no".'
)

# Default instruction when none provided by caller
DEFAULT_INSTRUCTION = "Retrieval relevant image or text with user's query"

vl_reranker_app = modal.App(
    "ai-workers-vl-reranker",
    secrets=[modal.Secret.from_name("worker-api-key")],
)


@vl_reranker_app.cls(
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
class VLRerankerServer:
    """VL reranker server for Qwen3-VL-Reranker-8B.

    Uses yes/no logit scoring with sigmoid for relevance probability.
    Supports multimodal inputs (text + image).
    """

    @modal.enter(snap=True)
    def load_models(self) -> None:
        """Load VL reranker model at container startup (snapshotted by GPU Memory Snapshot)."""
        import torch
        from loguru import logger
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.models: dict[str, object] = {}
        self.processors: dict[str, object] = {}
        self.yes_no_weights: dict[str, object] = {}

        for name, cfg in MODEL_CONFIGS.items():
            hf_id = cfg["hf_id"]
            registry_cfg = get_model(name)
            logger.info("Loading {} ...", hf_id)
            processor = AutoProcessor.from_pretrained(
                hf_id,
                trust_remote_code=registry_cfg.trust_remote_code,
                padding_side="left",
                cache_dir=HF_CACHE_DIR,
            )
            model = AutoModelForImageTextToText.from_pretrained(
                hf_id,
                torch_dtype=torch.float16,
                trust_remote_code=registry_cfg.trust_remote_code,
                device_map="auto",
                cache_dir=HF_CACHE_DIR,
            )
            model.eval()

            # Pre-extract yes/no weight vectors from lm_head for optimized scoring
            no_id = processor.tokenizer.convert_tokens_to_ids("no")
            yes_id = processor.tokenizer.convert_tokens_to_ids("yes")
            yes_no_weight = model.lm_head.weight.data[[no_id, yes_id], :].clone()

            self.models[name] = model
            self.processors[name] = processor
            self.yes_no_weights[name] = yes_no_weight
            logger.info(
                "Loaded {} successfully (yes_no_weight shape: {})", name, yes_no_weight.shape
            )

    def _score_pair(
        self,
        model_name: str,
        query: str,
        document: str,
        query_image: Any | None = None,
        document_image: Any | None = None,
        instruction: str | None = None,
    ) -> float:
        """Score a single query-document pair using optimized yes/no logit scoring.

        Uses backbone-only forward pass + pre-extracted yes/no weights from lm_head.
        Scoring: softmax([no, yes]) then take yes probability.
        Supports text-only and multimodal (image+text) query/document pairs.
        """
        import torch
        from torch.nn import functional as fn

        model = self.models[model_name]
        processor = self.processors[model_name]
        yes_no_weight = self.yes_no_weights[model_name]

        instruction = instruction or DEFAULT_INSTRUCTION

        # Build user content with optional images
        content_parts = []
        images = []

        # Query part
        if query_image:
            # Add a placeholder type string, Qwen-VL-utils will process the PIL images directly
            content_parts.append({"type": "image", "image": query_image})
            images.append(query_image)
        content_parts.append(
            {"type": "text", "text": f"<Instruct>: {instruction}\n<Query>: {query}"}
        )

        # Document part
        if document_image:
            content_parts.append({"type": "image", "image": document_image})
            images.append(document_image)
        content_parts.append({"type": "text", "text": f"\n<Document>: {document}"})

        messages = [
            {"role": "system", "content": RERANKER_SYSTEM_PROMPT},
            {"role": "user", "content": content_parts},
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        if images:
            inputs = processor(text=text, images=images, return_tensors="pt", padding=True).to(
                model.device
            )
        else:
            inputs = processor(text=text, return_tensors="pt", padding=True).to(model.device)

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

    @staticmethod
    def _load_image(url: str):
        """Load a PIL Image from URL with SSRF protection."""
        from ai_workers.common.utils import load_image_from_url

        return load_image_from_url(url)

    @modal.asgi_app()
    def serve(self):
        import asyncio

        from fastapi import Body, FastAPI, Request
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel, Field

        app = FastAPI(title="Qwen3 VL Reranker (8B)")

        class VLRerankDocument(BaseModel):
            text: str
            image_url: str | None = None

        class VLRerankRequest(BaseModel):
            model: str = "qwen3-vl-reranker-8b"
            query: str
            query_image_url: str | None = None
            documents: list[str] | list[VLRerankDocument] = Field(max_length=64)
            top_n: int | None = None

        class RerankResult(BaseModel):
            index: int
            relevance_score: float
            document: str

        class RerankResponse(BaseModel):
            model: str
            results: list[RerankResult]

        # Rebuild to resolve forward references (VLRerankDocument used in VLRerankRequest)
        VLRerankRequest.model_rebuild()

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

        @app.post("/v1/rerank", response_model=RerankResponse)
        async def rerank(body: VLRerankRequest = Body(...)):
            if body.model not in MODEL_CONFIGS:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": f"Unknown model: {body.model}. "
                        f"Available: {list(MODEL_CONFIGS.keys())}"
                    },
                )

            # Deduplicate image URLs
            image_urls = set()
            if body.query_image_url:
                image_urls.add(body.query_image_url)
            for doc in body.documents:
                if not isinstance(doc, str) and doc.image_url:
                    image_urls.add(doc.image_url)

            # Pre-fetch images concurrently
            unique_urls = list(image_urls)
            fetched_images = await asyncio.gather(
                *(asyncio.to_thread(self._load_image, url) for url in unique_urls)
            )
            image_map = dict(zip(unique_urls, fetched_images, strict=True))

            # Score each document against the query
            results = []
            for i, doc in enumerate(body.documents):
                if isinstance(doc, str):
                    doc_text = doc
                    doc_image = None
                else:
                    doc_text = doc.text
                    doc_image = doc.image_url

                score = self._score_pair(
                    body.model,
                    body.query,
                    doc_text,
                    query_image=image_map.get(body.query_image_url)
                    if body.query_image_url
                    else None,
                    document_image=image_map.get(doc_image) if doc_image else None,
                )
                results.append(RerankResult(index=i, relevance_score=score, document=doc_text))

            # Sort by relevance score descending
            results.sort(key=lambda x: x.relevance_score, reverse=True)

            # Apply top_n if specified
            if body.top_n is not None:
                results = results[: body.top_n]

            return RerankResponse(model=body.model, results=results)

        return app
