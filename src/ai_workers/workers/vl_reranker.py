"""Vision-Language Reranker worker using Custom FastAPI (merged light + heavy).

Serves both Qwen3-VL-Reranker-2B (light) and Qwen3-VL-Reranker-8B (heavy)
from a single A10G container. Routes by ``model`` field in request.

Both models loaded at startup (~20GB total on A10G 24GB VRAM).
Uses AutoModelForImageTextToText with yes/no logit scoring for relevance.

Supports text-only and image+text multimodal query/document pairs.

Models downloaded directly from HuggingFace Hub via Xet protocol
at container startup (~1GB/s). No R2 storage needed.
"""

import modal

from ai_workers.common.images import transformers_image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCALEDOWN_WINDOW = 300  # 5 minutes
KEEP_WARM = 0  # Scale to zero when idle

# Models served by this single app — loaded from HuggingFace Hub
MODEL_CONFIGS = {
    "qwen3-vl-reranker-2b": {"hf_id": "Qwen/Qwen3-VL-Reranker-2B"},
    "qwen3-vl-reranker-8b": {"hf_id": "Qwen/Qwen3-VL-Reranker-8B"},
}

# System prompt required by Qwen3-VL-Reranker
RERANKER_PREFIX = 'Judge whether the Document is relevant to the Query. Answer only "yes" or "no".'

vl_reranker_app = modal.App(
    "ai-workers-vl-reranker",
    secrets=[modal.Secret.from_name("worker-api-key")],
)


@vl_reranker_app.cls(
    gpu="A10G",
    image=transformers_image(),
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=KEEP_WARM,
    timeout=1800,
)
@modal.concurrent(max_inputs=100)
class VLRerankerServer:
    """Merged VL reranker server for Qwen3-VL-Reranker-2B + 8B.

    Both models loaded at startup. Routes request to correct model
    via the ``model`` field. Uses yes/no logit scoring with sigmoid
    for relevance probability. Supports multimodal inputs.
    """

    @modal.enter()
    def load_models(self) -> None:
        """Load both VL reranker models from HuggingFace Hub at container startup."""
        import torch
        from loguru import logger
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.models: dict[str, object] = {}
        self.processors: dict[str, object] = {}

        for name, cfg in MODEL_CONFIGS.items():
            hf_id = cfg["hf_id"]
            logger.info("Loading {} from HuggingFace Hub...", hf_id)
            processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
            model = AutoModelForImageTextToText.from_pretrained(
                hf_id,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto",
            )
            model.eval()
            self.models[name] = model
            self.processors[name] = processor
            logger.info("Loaded {} successfully", name)

    def _score_pair(
        self,
        model_name: str,
        query: str,
        document: str,
        query_image_url: str | None = None,
        document_image_url: str | None = None,
    ) -> float:
        """Score a single query-document pair using yes/no logit comparison.

        Supports text-only and multimodal (image+text) query/document pairs.
        """
        import torch

        model = self.models[model_name]
        processor = self.processors[model_name]

        # Build user content with optional images
        content_parts = []
        images = []

        # Query part
        if query_image_url:
            content_parts.append({"type": "image", "image": query_image_url})
            images.append(self._load_image(query_image_url))
        content_parts.append({"type": "text", "text": f"<Query>\n{query}\n</Query>"})

        # Document part
        if document_image_url:
            content_parts.append({"type": "image", "image": document_image_url})
            images.append(self._load_image(document_image_url))
        content_parts.append({"type": "text", "text": f"\n<Document>\n{document}\n</Document>"})

        messages = [
            {"role": "system", "content": RERANKER_PREFIX},
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
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits

            # Get logits for "yes" and "no" tokens
            yes_id = processor.tokenizer.convert_tokens_to_ids("yes")
            no_id = processor.tokenizer.convert_tokens_to_ids("no")
            yes_logit = logits[yes_id].float()
            no_logit = logits[no_id].float()

            # Sigmoid of (yes - no) gives relevance probability
            score = torch.sigmoid(yes_logit - no_logit).item()

        return score

    @staticmethod
    def _load_image(url: str):
        """Load a PIL Image from URL securely."""
        from ai_workers.common.utils import load_image_from_url

        return load_image_from_url(url)

    @modal.asgi_app()
    def serve(self):
        from fastapi import Body, FastAPI, Request
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel

        app = FastAPI(title="Qwen3 VL Reranker (Light + Heavy)")

        class VLRerankDocument(BaseModel):
            text: str
            image_url: str | None = None

        class VLRerankRequest(BaseModel):
            model: str = "qwen3-vl-reranker-2b"
            query: str
            query_image_url: str | None = None
            documents: list[str] | list[VLRerankDocument]
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
                    query_image_url=body.query_image_url,
                    document_image_url=doc_image,
                )
                results.append(RerankResult(index=i, relevance_score=score, document=doc_text))

            # Sort by relevance score descending
            results.sort(key=lambda x: x.relevance_score, reverse=True)

            # Apply top_n if specified
            if body.top_n is not None:
                results = results[: body.top_n]

            return RerankResponse(model=body.model, results=results)

        return app
