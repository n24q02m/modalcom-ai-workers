"""Vision-Language Embedding worker using Custom FastAPI (merged light + heavy).

Serves both Qwen3-VL-Embedding-2B (light) and Qwen3-VL-Embedding-8B (heavy)
from a single A10G container. Routes by ``model`` field in request.

Both models loaded at startup (~20GB total on A10G 24GB VRAM).
Uses AutoModel + AutoProcessor with mean pooling + L2 normalize.

Supports text-only and image+text multimodal inputs.

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
EMBEDDING_DIM = 1024

# Models served by this single app — loaded from HuggingFace Hub
MODEL_CONFIGS = {
    "qwen3-vl-embedding-2b": {"hf_id": "Qwen/Qwen3-VL-Embedding-2B"},
    "qwen3-vl-embedding-8b": {"hf_id": "Qwen/Qwen3-VL-Embedding-8B"},
}

vl_embedding_app = modal.App(
    "ai-workers-vl-embedding",
    secrets=[modal.Secret.from_name("worker-api-key")],
)


@vl_embedding_app.cls(
    gpu="A10G",
    image=transformers_image(),
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=KEEP_WARM,
    timeout=1800,
)
@modal.concurrent(max_inputs=100)
class VLEmbeddingServer:
    """Merged VL embedding server for Qwen3-VL-Embedding-2B + 8B.

    Both models loaded at startup. Routes request to correct model
    via the ``model`` field. Supports text-only and multimodal (image+text) inputs.
    """

    @modal.enter()
    def load_models(self) -> None:
        """Load both VL embedding models from HuggingFace Hub at container startup."""
        import torch
        from loguru import logger
        from transformers import AutoModel, AutoProcessor

        self.models: dict[str, object] = {}
        self.processors: dict[str, object] = {}

        for name, cfg in MODEL_CONFIGS.items():
            hf_id = cfg["hf_id"]
            logger.info("Loading {} from HuggingFace Hub...", hf_id)
            processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
            model = AutoModel.from_pretrained(
                hf_id,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto",
            )
            model.eval()
            self.models[name] = model
            self.processors[name] = processor
            logger.info("Loaded {} successfully", name)

    def _embed_text(self, model_name: str, texts: list[str]) -> list[list[float]]:
        """Embed text-only inputs using the specified model."""
        import torch

        model = self.models[model_name]
        processor = self.processors[model_name]

        # Wrap texts in chat format for VL model
        messages_batch = [
            [{"role": "user", "content": [{"type": "text", "text": t}]}] for t in texts
        ]

        # Process each message separately (VL processor handles one at a time)
        all_embeddings = []
        for messages in messages_batch:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(text=text, return_tensors="pt", padding=True).to(model.device)

            with torch.no_grad():
                outputs = model(**inputs)
                # Mean pooling over sequence dimension
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                token_embeddings = outputs.last_hidden_state
                masked = token_embeddings * attention_mask
                summed = masked.sum(dim=1)
                counts = attention_mask.sum(dim=1).clamp(min=1e-9)
                embedding = summed / counts
                # L2 normalize
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

            all_embeddings.append(embedding[:, :EMBEDDING_DIM].cpu().tolist()[0])

        return all_embeddings

    def _embed_multimodal(self, model_name: str, text: str, image_url: str) -> list[float]:
        """Embed a single image+text pair."""
        import torch

        from ai_workers.common.utils import load_image_from_url

        model = self.models[model_name]
        processor = self.processors[model_name]

        # Load image from URL securely
        image = load_image_from_url(image_url)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            }
        ]

        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=text_input, images=[image], return_tensors="pt", padding=True).to(
            model.device
        )

        with torch.no_grad():
            outputs = model(**inputs)
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            token_embeddings = outputs.last_hidden_state
            masked = token_embeddings * attention_mask
            summed = masked.sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1e-9)
            embedding = summed / counts
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        return embedding[:, :EMBEDDING_DIM].cpu().tolist()[0]

    @modal.asgi_app()
    def serve(self):
        from fastapi import Body, FastAPI, Request
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel

        app = FastAPI(title="Qwen3 VL Embedding (Light + Heavy)")

        class VLEmbeddingInput(BaseModel):
            text: str
            image_url: str | None = None

        class VLEmbeddingRequest(BaseModel):
            model: str = "qwen3-vl-embedding-2b"
            input: str | list[str] | VLEmbeddingInput | list[VLEmbeddingInput]
            encoding_format: str = "float"

        class EmbeddingData(BaseModel):
            object: str = "embedding"
            embedding: list[float]
            index: int

        class EmbeddingResponse(BaseModel):
            object: str = "list"
            data: list[EmbeddingData]
            model: str

        # Rebuild to resolve forward references (VLEmbeddingInput used in VLEmbeddingRequest)
        VLEmbeddingRequest.model_rebuild()

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

        @app.post("/v1/embeddings", response_model=EmbeddingResponse)
        async def create_embeddings(body: VLEmbeddingRequest = Body(...)):
            if body.model not in MODEL_CONFIGS:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": f"Unknown model: {body.model}. "
                        f"Available: {list(MODEL_CONFIGS.keys())}"
                    },
                )

            embeddings: list[list[float]] = []

            if isinstance(body.input, str):
                # Single text input
                embeddings = self._embed_text(body.model, [body.input])
            elif isinstance(body.input, list) and body.input and isinstance(body.input[0], str):
                # List of text inputs
                embeddings = self._embed_text(body.model, body.input)
            elif isinstance(body.input, VLEmbeddingInput):
                # Single multimodal input
                if body.input.image_url:
                    emb = self._embed_multimodal(body.model, body.input.text, body.input.image_url)
                    embeddings = [emb]
                else:
                    embeddings = self._embed_text(body.model, [body.input.text])
            elif isinstance(body.input, list):
                # List of multimodal inputs
                for item in body.input:
                    if isinstance(item, VLEmbeddingInput) and item.image_url:
                        emb = self._embed_multimodal(body.model, item.text, item.image_url)
                        embeddings.append(emb)
                    elif isinstance(item, VLEmbeddingInput):
                        embs = self._embed_text(body.model, [item.text])
                        embeddings.extend(embs)

            data = [EmbeddingData(embedding=emb, index=i) for i, emb in enumerate(embeddings)]
            return EmbeddingResponse(data=data, model=body.model)

        return app
