"""Text Embedding workers using vLLM.

vLLM natively serves OpenAI-compatible /v1/embeddings endpoint.
Two apps: embedding_light (0.6B, T4) and embedding_heavy (8B, A10G).

LiteLLM integration:
  model: openai/qwen3-embedding-0.6b
  api_base: https://<modal-url>
"""

from __future__ import annotations

import modal

from ai_workers.common.images import MODELS_MOUNT_PATH, vllm_image
from ai_workers.common.r2 import get_modal_cloud_bucket_mount

# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------

SCALEDOWN_WINDOW = 300  # 5 minutes
KEEP_WARM = 0  # Scale to zero when idle

r2_mount = get_modal_cloud_bucket_mount()

# ---------------------------------------------------------------------------
# Embedding Light (Qwen3-Embedding-0.6B, T4)
# ---------------------------------------------------------------------------

embedding_light_app = modal.App(
    "ai-workers-qwen3-embedding-0.6b",
    secrets=[modal.Secret.from_name("r2-credentials"), modal.Secret.from_name("worker-api-key")],
)

MODEL_LIGHT = "qwen3-embedding-0.6b"
EMBEDDING_DIM = 1024


@embedding_light_app.cls(
    gpu="T4",
    image=vllm_image(),
    volumes={MODELS_MOUNT_PATH: r2_mount},
    scaledown_window=SCALEDOWN_WINDOW,
    keep_warm=KEEP_WARM,
    timeout=600,
    allow_concurrent_inputs=100,
)
class EmbeddingLightServer:
    """vLLM-based embedding server for Qwen3-Embedding-0.6B."""

    @modal.enter()
    def start_engine(self) -> None:
        """Initialize vLLM engine at container startup."""
        from vllm import LLM  # type: ignore

        model_path = f"{MODELS_MOUNT_PATH}/{MODEL_LIGHT}"
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

        from fastapi import FastAPI
        from pydantic import BaseModel

        app = FastAPI(title="Qwen3 Embedding Light")

        class EmbeddingRequest(BaseModel):
            model: str = MODEL_LIGHT
            input: str | list[str]
            encoding_format: str = "float"

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
            return {"status": "ok", "model": MODEL_LIGHT}

        @app.post("/v1/embeddings", response_model=EmbeddingResponse)
        async def create_embeddings(request: EmbeddingRequest):
            texts = request.input if isinstance(request.input, list) else [request.input]

            outputs = self.engine.embed(texts)

            data = []
            total_tokens = 0
            for i, output in enumerate(outputs):
                embedding = output.outputs.embedding[:EMBEDDING_DIM]
                data.append(EmbeddingData(embedding=embedding, index=i))
                total_tokens += len(output.prompt_token_ids)

            return EmbeddingResponse(
                data=data,
                model=request.model,
                usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens},
            )

        return app


# ---------------------------------------------------------------------------
# Embedding Heavy (Qwen3-Embedding-8B, A10G)
# ---------------------------------------------------------------------------

embedding_heavy_app = modal.App(
    "ai-workers-qwen3-embedding-8b",
    secrets=[modal.Secret.from_name("r2-credentials"), modal.Secret.from_name("worker-api-key")],
)

MODEL_HEAVY = "qwen3-embedding-8b"


@embedding_heavy_app.cls(
    gpu="A10G",
    image=vllm_image(),
    volumes={MODELS_MOUNT_PATH: r2_mount},
    scaledown_window=SCALEDOWN_WINDOW,
    keep_warm=KEEP_WARM,
    timeout=600,
    allow_concurrent_inputs=100,
)
class EmbeddingHeavyServer:
    """vLLM-based embedding server for Qwen3-Embedding-8B."""

    @modal.enter()
    def start_engine(self) -> None:
        from vllm import LLM  # type: ignore

        model_path = f"{MODELS_MOUNT_PATH}/{MODEL_HEAVY}"
        self.engine = LLM(
            model=model_path,
            task="embed",
            dtype="float16",
            trust_remote_code=True,
            max_model_len=8192,
            enforce_eager=True,
        )

    @modal.asgi_app()
    def serve(self):

        from fastapi import FastAPI
        from pydantic import BaseModel

        app = FastAPI(title="Qwen3 Embedding Heavy")

        class EmbeddingRequest(BaseModel):
            model: str = MODEL_HEAVY
            input: str | list[str]
            encoding_format: str = "float"

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
            return {"status": "ok", "model": MODEL_HEAVY}

        @app.post("/v1/embeddings", response_model=EmbeddingResponse)
        async def create_embeddings(request: EmbeddingRequest):
            texts = request.input if isinstance(request.input, list) else [request.input]

            outputs = self.engine.embed(texts)

            data = []
            total_tokens = 0
            for i, output in enumerate(outputs):
                embedding = output.outputs.embedding[:EMBEDDING_DIM]
                data.append(EmbeddingData(embedding=embedding, index=i))
                total_tokens += len(output.prompt_token_ids)

            return EmbeddingResponse(
                data=data,
                model=request.model,
                usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens},
            )

        return app
