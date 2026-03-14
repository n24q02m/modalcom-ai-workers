"""Text Embedding worker using Custom FastAPI (merged light + heavy).

Serves both Qwen3-Embedding-0.6B (light) and Qwen3-Embedding-8B (heavy)
from a single A10G container. Routes by ``model`` field in request.

Both models loaded at startup (~17GB total on A10G 24GB VRAM).
Custom transformers forward pass + mean pooling + L2 normalize.

Uses Modal Volume (pre-downloaded weights) + GPU Memory Snapshot
for fast cold start (~5-10s instead of >10 minutes).

LiteLLM integration:
  model: openai/qwen3-embedding-0.6b  (or qwen3-embedding-8b)
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
EMBEDDING_DIM = 1024

# Models served by this single app — loaded from HuggingFace Hub
MODEL_CONFIGS = {
    "qwen3-embedding-0.6b": {"hf_id": "Qwen/Qwen3-Embedding-0.6B"},
    "qwen3-embedding-8b": {"hf_id": "Qwen/Qwen3-Embedding-8B"},
}

embedding_app = modal.App(
    "ai-workers-embedding",
    secrets=[modal.Secret.from_name("worker-api-key")],
)


@embedding_app.cls(
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
class EmbeddingServer:
    """Merged embedding server for Qwen3-Embedding-0.6B + 8B.

    Both models loaded at startup. Routes request to correct model
    via the ``model`` field. Uses custom transformers forward pass
    with mean pooling and L2 normalization (no vLLM needed for embeddings).
    """

    @modal.enter(snap=True)
    def load_models(self) -> None:
        """Load both embedding models at container startup (snapshotted by GPU Memory Snapshot)."""
        import torch
        from loguru import logger
        from transformers import AutoModel, AutoTokenizer

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
            model = AutoModel.from_pretrained(
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

    def _embed(self, model_name: str, texts: list[str]) -> list[list[float]]:
        """Embed texts using the specified model."""
        import torch

        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling over sequence dimension (ignore padding)
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            token_embeddings = outputs.last_hidden_state
            masked = token_embeddings * attention_mask
            summed = masked.sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1e-9)
            embeddings = summed / counts
            # L2 normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings[:, :EMBEDDING_DIM].cpu().tolist()

    @modal.asgi_app()
    def serve(self):
        from fastapi import Body, FastAPI
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel

        app = FastAPI(title="Qwen3 Embedding (Light + Heavy)")

        class EmbeddingRequest(BaseModel):
            model: str = "qwen3-embedding-0.6b"
            input: str | list[str]
            encoding_format: str | None = "float"

        class EmbeddingData(BaseModel):
            object: str = "embedding"
            embedding: list[float]
            index: int

        class EmbeddingResponse(BaseModel):
            object: str = "list"
            data: list[EmbeddingData]
            model: str
            usage: dict[str, int]

        from ai_workers.common.auth import auth_middleware

        app.middleware("http")(auth_middleware)

        @app.get("/health")
        async def health():
            return {
                "status": "ok",
                "models": list(MODEL_CONFIGS.keys()),
            }

        @app.post("/v1/embeddings", response_model=EmbeddingResponse)
        async def create_embeddings(body: EmbeddingRequest = Body(...)):
            if body.model not in MODEL_CONFIGS:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": f"Unknown model: {body.model}. "
                        f"Available: {list(MODEL_CONFIGS.keys())}"
                    },
                )

            texts = body.input if isinstance(body.input, list) else [body.input]
            embeddings = self._embed(body.model, texts)

            data = [EmbeddingData(embedding=emb, index=i) for i, emb in enumerate(embeddings)]
            total_tokens = sum(len(self.tokenizers[body.model].encode(t)) for t in texts)

            return EmbeddingResponse(
                data=data,
                model=body.model,
                usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens},
            )

        return app
