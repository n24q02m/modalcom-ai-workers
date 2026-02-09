"""Text Embedding workers using vLLM.

vLLM natively serves OpenAI-compatible /v1/embeddings endpoint.
Two apps: embedding_light (0.6B, T4) and embedding_heavy (8B, A10G).

LiteLLM integration:
  model: openai/qwen3-embedding-0.6b
  api_base: https://<modal-url>
"""

from __future__ import annotations

import modal
from fastapi import FastAPI, Request
from pydantic import BaseModel

from ai_workers.common.images import MODELS_MOUNT_PATH, vllm_image
from ai_workers.common.r2 import get_modal_cloud_bucket_mount

# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------

SCALEDOWN_WINDOW = 300  # 5 minutes
KEEP_WARM = 0  # Scale to zero when idle

r2_mount = get_modal_cloud_bucket_mount()

MODEL_LIGHT = "qwen3-embedding-0.6b"
MODEL_HEAVY = "qwen3-embedding-8b"
EMBEDDING_DIM = 1024


# ---------------------------------------------------------------------------
# Shared Data Models
# ---------------------------------------------------------------------------


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: list[float]
    index: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: dict[str, int]


class BaseEmbeddingRequest(BaseModel):
    input: str | list[str]
    encoding_format: str = "float"


def create_embedding_app(
    server_instance, model_name: str, embedding_dim: int, app_title: str
) -> FastAPI:
    """Factory function to create the FastAPI app for embedding workers."""

    app = FastAPI(title=app_title)

    class EmbeddingRequest(BaseEmbeddingRequest):
        model: str = model_name

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        if request.url.path in ("/health", "/"):
            return await call_next(request)
        from ai_workers.common.auth import verify_api_key

        await verify_api_key(request)
        return await call_next(request)

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": model_name}

    @app.post("/v1/embeddings", response_model=EmbeddingResponse)
    async def create_embeddings(request: EmbeddingRequest):
        texts = request.input if isinstance(request.input, list) else [request.input]

        outputs = server_instance.engine.embed(texts)

        data = []
        total_tokens = 0
        for i, output in enumerate(outputs):
            embedding = output.outputs.embedding[:embedding_dim]
            data.append(EmbeddingData(embedding=embedding, index=i))
            total_tokens += len(output.prompt_token_ids)

        return EmbeddingResponse(
            data=data,
            model=request.model,
            usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens},
        )

    return app


class BaseEmbeddingServer:
    """Base class for vLLM-based embedding servers."""

    model_name: str
    embedding_dim: int
    app_title: str

    @modal.enter()
    def start_engine(self) -> None:
        """Initialize vLLM engine at container startup."""
        from vllm import LLM

        model_path = f"{MODELS_MOUNT_PATH}/{self.model_name}"
        self.engine = LLM(
            model=model_path,
            task="embed",
            dtype="float16",
            trust_remote_code=True,
            max_model_len=8192,
            enforce_eager=True,  # Avoid CUDA graph overhead for embedding
        )

    @modal.asgi_app()
    def serve(self):
        """Expose OpenAI-compatible /v1/embeddings endpoint."""
        return create_embedding_app(self, self.model_name, self.embedding_dim, self.app_title)


# ---------------------------------------------------------------------------
# Embedding Light (Qwen3-Embedding-0.6B, T4)
# ---------------------------------------------------------------------------

embedding_light_app = modal.App(
    "ai-workers-qwen3-embedding-0.6b",
    secrets=[modal.Secret.from_name("r2-credentials"), modal.Secret.from_name("worker-api-key")],
)


@embedding_light_app.cls(
    gpu="T4",
    image=vllm_image(),
    volumes={MODELS_MOUNT_PATH: r2_mount},
    scaledown_window=SCALEDOWN_WINDOW,
    keep_warm=KEEP_WARM,
    timeout=600,
    allow_concurrent_inputs=100,
)
class EmbeddingLightServer(BaseEmbeddingServer):
    """vLLM-based embedding server for Qwen3-Embedding-0.6B."""

    model_name = MODEL_LIGHT
    embedding_dim = EMBEDDING_DIM
    app_title = "Qwen3 Embedding Light"


# ---------------------------------------------------------------------------
# Embedding Heavy (Qwen3-Embedding-8B, A10G)
# ---------------------------------------------------------------------------

embedding_heavy_app = modal.App(
    "ai-workers-qwen3-embedding-8b",
    secrets=[modal.Secret.from_name("r2-credentials"), modal.Secret.from_name("worker-api-key")],
)


@embedding_heavy_app.cls(
    gpu="A10G",
    image=vllm_image(),
    volumes={MODELS_MOUNT_PATH: r2_mount},
    scaledown_window=SCALEDOWN_WINDOW,
    keep_warm=KEEP_WARM,
    timeout=600,
    allow_concurrent_inputs=100,
)
class EmbeddingHeavyServer(BaseEmbeddingServer):
    """vLLM-based embedding server for Qwen3-Embedding-8B."""

    model_name = MODEL_HEAVY
    embedding_dim = EMBEDDING_DIM
    app_title = "Qwen3 Embedding Heavy"
