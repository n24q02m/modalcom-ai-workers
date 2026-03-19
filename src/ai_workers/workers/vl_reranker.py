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

import modal

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
            logger.info("Loading {} ...", hf_id)
            processor = AutoProcessor.from_pretrained(
                hf_id,
                trust_remote_code=True,
                padding_side="left",
                cache_dir=HF_CACHE_DIR,
            )
            model = AutoModelForImageTextToText.from_pretrained(
                hf_id,
                torch_dtype=torch.float16,
                trust_remote_code=True,
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

    def _score_batch(
        self,
        model_name: str,
        query: str,
        documents: list[str],
        query_image_url: str | None = None,
        document_image_urls: list[str | None] | None = None,
        instruction: str | None = None,
    ) -> list[float]:
        """Score a batch of query-document pairs using optimized chunked inference.

        Batched alternative to `_score_pair` to prevent sequential processing bottleneck.
        Processes in chunks (e.g., batch_size=32) with padding.
        Uses right padding and extracts the last token hidden state via attention_mask sum.
        Supports text-only and multimodal (image+text) query/document pairs.
        """
        import torch
        from torch.nn import functional as fn

        model = self.models[model_name]
        processor = self.processors[model_name]
        yes_no_weight = self.yes_no_weights[model_name]

        instruction = instruction or DEFAULT_INSTRUCTION

        # Ensure right padding for correct last-token extraction
        processor.tokenizer.padding_side = "right"
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

        # Load query image once if provided
        query_image = None
        if query_image_url:
            query_image = self._load_image(query_image_url)

        if document_image_urls is None:
            document_image_urls = [None] * len(documents)

        all_scores = []
        batch_size = 32

        for i in range(0, len(documents), batch_size):
            chunk_docs = documents[i : i + batch_size]
            chunk_doc_image_urls = document_image_urls[i : i + batch_size]

            messages_chunk = []
            images_chunk = []

            for doc, doc_image_url in zip(chunk_docs, chunk_doc_image_urls, strict=True):
                content_parts = []

                # Query part
                if query_image_url:
                    content_parts.append({"type": "image", "image": query_image_url})
                    images_chunk.append(query_image)
                content_parts.append(
                    {"type": "text", "text": f"<Instruct>: {instruction}\n<Query>: {query}"}
                )

                # Document part
                if doc_image_url:
                    content_parts.append({"type": "image", "image": doc_image_url})
                    images_chunk.append(self._load_image(doc_image_url))
                content_parts.append({"type": "text", "text": f"\n<Document>: {doc}"})

                messages_chunk.append(
                    [
                        {"role": "system", "content": RERANKER_SYSTEM_PROMPT},
                        {"role": "user", "content": content_parts},
                    ]
                )

            texts = [
                processor.apply_chat_template(
                    m,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for m in messages_chunk
            ]

            if images_chunk:
                inputs = processor(
                    text=texts, images=images_chunk, return_tensors="pt", padding=True
                ).to(model.device)
            else:
                inputs = processor(text=texts, return_tensors="pt", padding=True).to(model.device)

            with torch.no_grad():
                outputs = model.model(**inputs)

                # Find the last non-padding token indices
                last_token_indices = inputs["attention_mask"].sum(1) - 1
                curr_batch_size = len(chunk_docs)

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

    @staticmethod
    def _load_image(url: str):
        """Load a PIL Image from URL with SSRF protection."""
        from ai_workers.common.utils import load_image_from_url

        return load_image_from_url(url)

    @modal.asgi_app()
    def serve(self):
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

            # Extract text and optional image URLs for batch scoring
            doc_texts = []
            doc_images = []
            for doc in body.documents:
                if isinstance(doc, str):
                    doc_texts.append(doc)
                    doc_images.append(None)
                else:
                    doc_texts.append(doc.text)
                    doc_images.append(doc.image_url)

            # Score all documents against the query in batches
            scores = self._score_batch(
                model_name=body.model,
                query=body.query,
                documents=doc_texts,
                query_image_url=body.query_image_url,
                document_image_urls=doc_images,
            )

            results = []
            for i, (doc_text, score) in enumerate(zip(doc_texts, scores, strict=True)):
                results.append(RerankResult(index=i, relevance_score=score, document=doc_text))

            # Sort by relevance score descending
            results.sort(key=lambda x: x.relevance_score, reverse=True)

            # Apply top_n if specified
            if body.top_n is not None:
                results = results[: body.top_n]

            return RerankResponse(model=body.model, results=results)

        return app
