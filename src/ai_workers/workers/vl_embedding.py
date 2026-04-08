"""Vision-Language Embedding worker using Custom FastAPI (merged light + heavy).

Serves both Qwen3-VL-Embedding-2B (light) and Qwen3-VL-Embedding-8B (heavy)
from a single A10G container. Routes by ``model`` field in request.

Both models loaded at startup (~20GB total on A10G 24GB VRAM).
Uses official Qwen3-VL-Embedding approach: EOS token pooling + L2 normalize.
System instruction: "Represent the user's input."

Supports text-only and image+text multimodal inputs.

Uses Modal Volume (pre-downloaded weights) + GPU Memory Snapshot
for fast cold start (~5-10s instead of >10 minutes).
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
EMBEDDING_DIM = 1024

# Official system instruction for Qwen3-VL-Embedding
DEFAULT_INSTRUCTION = "Represent the user's input."

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
    volumes={HF_CACHE_DIR: hf_cache_vol},
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=KEEP_WARM,
    timeout=1800,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=100)
class VLEmbeddingServer:
    """Merged VL embedding server for Qwen3-VL-Embedding-2B + 8B.

    Both models loaded at startup. Routes request to correct model
    via the ``model`` field. Uses official Qwen3-VL-Embedding approach:
    EOS token pooling + L2 normalization. Supports text-only and multimodal inputs.
    """

    @modal.enter(snap=True)
    def load_models(self) -> None:
        """Load both VL embedding models at container startup (snapshotted by GPU Memory Snapshot)."""
        import torch
        from loguru import logger
        from transformers import AutoModel, AutoProcessor

        self.models: dict[str, object] = {}
        self.processors: dict[str, object] = {}

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
            model = AutoModel.from_pretrained(
                hf_id,
                torch_dtype=torch.float16,
                trust_remote_code=registry_cfg.trust_remote_code,
                device_map="auto",
                cache_dir=HF_CACHE_DIR,
            )
            model.eval()
            self.models[name] = model
            self.processors[name] = processor
            logger.info("Loaded {} successfully", name)

    @staticmethod
    def _last_token_pool(last_hidden_states, attention_mask):
        """Official Qwen3-VL-Embedding pooling: extract EOS token hidden state."""
        import torch

        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]

    def _embed_text(self, model_name: str, texts: list[str]) -> list[list[float]]:
        """Embed text-only inputs using EOS token pooling."""
        import torch

        model = self.models[model_name]
        processor = self.processors[model_name]

        # Wrap texts in chat format with system instruction
        messages_batch = [
            [
                {"role": "system", "content": [{"type": "text", "text": DEFAULT_INSTRUCTION}]},
                {"role": "user", "content": [{"type": "text", "text": t}]},
            ]
            for t in texts
        ]

        # Batched processing
        text_inputs = processor.apply_chat_template(
            messages_batch, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=text_inputs, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Official Qwen3-VL-Embedding: EOS token pooling
            embeddings = self._last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
            # L2 normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings[:, :EMBEDDING_DIM].cpu().tolist()

    def _embed_multimodal(
        self, model_name: str, texts: list[str], image_urls: list[str]
    ) -> list[list[float]]:
        """Embed a batch of image+text pairs with EOS token pooling."""
        import torch
        from qwen_vl_utils import process_vision_info

        from ai_workers.common.utils import is_safe_url

        # Validate URLs before passing to process_vision_info (SSRF protection)
        for url in image_urls:
            if not url.startswith("data:") and not is_safe_url(url):
                raise ValueError(f"URL blocked by SSRF protection: {url}")

        model = self.models[model_name]
        processor = self.processors[model_name]

        messages_batch = [
            [
                {"role": "system", "content": [{"type": "text", "text": DEFAULT_INSTRUCTION}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": url},
                        {"type": "text", "text": text},
                    ],
                },
            ]
            for text, url in zip(texts, image_urls, strict=False)
        ]

        text_inputs = processor.apply_chat_template(
            messages_batch, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages_batch)

        inputs = processor(
            text=text_inputs,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Official Qwen3-VL-Embedding: EOS token pooling
            embeddings = self._last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings[:, :EMBEDDING_DIM].cpu().tolist()

    @modal.asgi_app()
    def serve(self):
        import asyncio

        from fastapi import Body, FastAPI, Request
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel, field_validator

        app = FastAPI(title="Qwen3 VL Embedding (Light + Heavy)")

        max_input_length = 64

        class VLEmbeddingInput(BaseModel):
            text: str
            image_url: str | None = None

        class VLEmbeddingRequest(BaseModel):
            model: str = "qwen3-vl-embedding-2b"
            input: str | list[str] | VLEmbeddingInput | list[VLEmbeddingInput]
            encoding_format: str = "float"

            @field_validator("input")
            @classmethod
            def validate_input_length(cls, v):
                if isinstance(v, list) and len(v) > max_input_length:
                    msg = (
                        f"Input list too long ({len(v)} items). "
                        f"Maximum allowed: {max_input_length}."
                    )
                    raise ValueError(msg)
                return v

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
                embeddings = await asyncio.to_thread(self._embed_text, body.model, [body.input])
            elif isinstance(body.input, list) and body.input and isinstance(body.input[0], str):
                # List of text inputs
                embeddings = await asyncio.to_thread(self._embed_text, body.model, body.input)
            elif isinstance(body.input, VLEmbeddingInput):
                # Single multimodal input
                if body.input.image_url:
                    embeddings = await asyncio.to_thread(
                        self._embed_multimodal,
                        body.model, [body.input.text], [body.input.image_url]
                    )
                else:
                    embeddings = await asyncio.to_thread(self._embed_text, body.model, [body.input.text])
            elif isinstance(body.input, list):
                # List of multimodal inputs - aggregate for batching
                mm_indices = []
                mm_texts = []
                mm_urls = []
                text_indices = []
                text_texts = []

                for i, item in enumerate(body.input):
                    if isinstance(item, VLEmbeddingInput) and item.image_url:
                        mm_indices.append(i)
                        mm_texts.append(item.text)
                        mm_urls.append(item.image_url)
                    elif isinstance(item, VLEmbeddingInput):
                        text_indices.append(i)
                        text_texts.append(item.text)

                embeddings: list[list[float] | None] = [None] * len(body.input)

                if mm_indices:
                    mm_results = await asyncio.to_thread(self._embed_multimodal, body.model, mm_texts, mm_urls)
                    for idx, res in zip(mm_indices, mm_results, strict=False):
                        embeddings[idx] = res

                if text_indices:
                    text_results = await asyncio.to_thread(self._embed_text, body.model, text_texts)
                    for idx, res in zip(text_indices, text_results, strict=False):
                        embeddings[idx] = res

            data = [EmbeddingData(embedding=emb, index=i) for i, emb in enumerate(embeddings)]
            return EmbeddingResponse(data=data, model=body.model)

        return app
