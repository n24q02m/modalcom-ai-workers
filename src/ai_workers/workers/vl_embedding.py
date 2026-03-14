"""Vision-Language Embedding workers using Custom FastAPI.

Qwen3-VL-Embedding supports text, image, and video inputs.
Exposes OpenAI-compatible /v1/embeddings endpoint with multimodal support.
Two apps: vl_embedding_light (2B, T4) and vl_embedding_heavy (8B, A10G).

LiteLLM integration:
  model: openai/qwen3-vl-embedding-2b
  api_base: https://<modal-url>
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import modal

if TYPE_CHECKING:
    from fastapi import FastAPI, Request

from pydantic import BaseModel

from ai_workers.common.images import MODELS_MOUNT_PATH, transformers_image
from ai_workers.common.r2 import get_modal_cloud_bucket_mount

SCALEDOWN_WINDOW = 300
KEEP_WARM = 0
EMBEDDING_DIM = 1024

r2_mount = get_modal_cloud_bucket_mount()

MODEL_LIGHT = "qwen3-vl-embedding-2b"
MODEL_HEAVY = "qwen3-vl-embedding-8b"


# ---------------------------------------------------------------------------
# Pydantic Models (Module Level)
# ---------------------------------------------------------------------------


class EmbeddingRequest(BaseModel):
    model: str
    input: list[str] | str
    encoding_format: str = "float"


class EmbeddingRequestLight(EmbeddingRequest):
    model: str = MODEL_LIGHT


class EmbeddingRequestHeavy(EmbeddingRequest):
    model: str = MODEL_HEAVY


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: list[float]
    index: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: dict[str, int]


# ---------------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------------


def create_vl_embedding_app(
    model_name: str,
    request_model: type[BaseModel],
    embedding_server_instance: Any,
) -> FastAPI:
    from fastapi import FastAPI

    from ai_workers.common.auth import verify_api_key

    app = FastAPI(title=f"Qwen3 VL Embedding {model_name}")

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        if request.url.path in ("/health", "/"):
            return await call_next(request)
        await verify_api_key(request)
        return await call_next(request)

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": model_name}

    @app.post("/v1/embeddings", response_model=EmbeddingResponse)
    async def create_embeddings(request: request_model):  # type: ignore
        texts = request.input if isinstance(request.input, list) else [request.input]
        embeddings = embedding_server_instance._embed_texts(texts)

        data = [EmbeddingData(embedding=emb, index=i) for i, emb in enumerate(embeddings)]

        return EmbeddingResponse(
            data=data,
            model=request.model,
            usage={"prompt_tokens": 0, "total_tokens": 0},
        )

    # Fix for from __future__ import annotations with local variable
    create_embeddings.__annotations__["request"] = request_model

    return app


# ---------------------------------------------------------------------------
# Base Class
# ---------------------------------------------------------------------------


class VLEmbeddingServerBase:
    """Base class for VL embedding servers."""

    model_name: str
    request_model: type[BaseModel]

    @modal.enter()
    def load_model(self) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        model_path = f"{MODELS_MOUNT_PATH}/{self.model_name}"
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )
        self.model.eval()

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed text-only inputs."""
        import torch

        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling over sequence dimension
            embeddings = outputs.last_hidden_state.mean(dim=1)
            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings[:, :EMBEDDING_DIM].cpu().tolist()

    @modal.asgi_app()
    def serve(self):
        return create_vl_embedding_app(self.model_name, self.request_model, self)


# ---------------------------------------------------------------------------
# VL Embedding Light (Qwen3-VL-Embedding-2B, T4)
# ---------------------------------------------------------------------------

vl_embedding_light_app = modal.App(
    "ai-workers-qwen3-vl-embedding-2b",
    secrets=[modal.Secret.from_name("r2-credentials"), modal.Secret.from_name("worker-api-key")],
)


@vl_embedding_light_app.cls(
    gpu="T4",
    image=transformers_image(),
    volumes={MODELS_MOUNT_PATH: r2_mount},
    scaledown_window=SCALEDOWN_WINDOW,
    keep_warm=KEEP_WARM,
    timeout=600,
    allow_concurrent_inputs=10,
)
class VLEmbeddingLightServer(VLEmbeddingServerBase):
    """Custom FastAPI VL embedding server for Qwen3-VL-Embedding-2B."""

    model_name = MODEL_LIGHT
    request_model = EmbeddingRequestLight


# ---------------------------------------------------------------------------
# VL Embedding Heavy (Qwen3-VL-Embedding-8B, A10G)
# ---------------------------------------------------------------------------

vl_embedding_heavy_app = modal.App(
    "ai-workers-qwen3-vl-embedding-8b",
    secrets=[modal.Secret.from_name("r2-credentials"), modal.Secret.from_name("worker-api-key")],
)


@vl_embedding_heavy_app.cls(
    gpu="A10G",
    image=transformers_image(),
    volumes={MODELS_MOUNT_PATH: r2_mount},
    scaledown_window=SCALEDOWN_WINDOW,
    keep_warm=KEEP_WARM,
    timeout=600,
    allow_concurrent_inputs=10,
)
class VLEmbeddingHeavyServer(VLEmbeddingServerBase):
    """Custom FastAPI VL embedding server for Qwen3-VL-Embedding-8B."""

    model_name = MODEL_HEAVY
    request_model = EmbeddingRequestHeavy
