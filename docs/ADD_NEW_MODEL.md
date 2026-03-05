# Adding a New Model

This document describes the steps to add a new AI model to the workers system.

## Overview

```
1. Register in Model Registry (config.py)
2. Create Worker module (workers/*.py)
3. Deploy to Modal
4. Add to LiteLLM config
5. Test endpoint
```

## Step 1: Register in Model Registry

File: `src/ai_workers/common/config.py`

Add a new entry to the Model Registry:

```python
_register(
    ModelConfig(
        # Unique name, used as key
        name="my-new-model-7b",
        # HuggingFace model ID (used for download/convert)
        hf_id="organization/MyNewModel-7B",
        # Task: EMBEDDING, RERANKER_LLM, VL_EMBEDDING, VL_RERANKER, OCR, ASR
        task=Task.EMBEDDING,
        # Tier: LIGHT or HEAVY
        tier=Tier.HEAVY,
        # Precision: FP16 (default) or BF16
        # Only use BF16 when the model was trained with BF16 and cannot be converted to FP16
        precision=Precision.FP16,
        # GPU: T4 (models <= 2B) or A10G (models > 2B or BF16)
        gpu=GPU.A10G,
        # Serving: VLLM (text embedding only) or CUSTOM_FASTAPI
        serving_engine=ServingEngine.VLLM,
        # Model class: AUTO_MODEL, CAUSAL_LM, SEQ2SEQ
        model_class=ModelClassType.AUTO_MODEL,
        # Trust remote code (True for Qwen, DeepSeek, ...)
        trust_remote_code=True,
        # Module path of the worker file
        worker_module="ai_workers.workers.my_new_worker",
        # Extra kwargs for model loading (optional)
        extra_load_kwargs={},
    )
)
```

### Choosing a GPU

| Model Size | Precision | GPU  | Reason                    |
|-----------|-----------|------|---------------------------|
| <= 2B     | FP16      | T4   | Sufficient VRAM (16GB)    |
| 3-8B      | FP16      | A10G | Requires 24GB VRAM        |
| Any       | BF16      | A10G+| T4 does not support BF16  |

### Choosing Precision

- **FP16** (default): Suitable for most models. Smaller, faster.
- **BF16**: Only when the model was trained with BF16 and converting to FP16 causes quality degradation (e.g., DeepSeek-OCR-2).

## Step 2: Create Worker Module

File: `src/ai_workers/workers/my_new_worker.py`

### Basic Template

```python
"""My New Worker description."""

from __future__ import annotations

import modal

from ai_workers.common.images import transformers_image

SCALEDOWN_WINDOW = 300
KEEP_WARM = 0
MODEL_NAME = "my-new-model-7b"

my_app = modal.App(
    "ai-workers-my-new-model-7b",
    secrets=[
        modal.Secret.from_name("worker-api-key"),
    ],
)


@my_app.cls(
    gpu="A10G",
    image=transformers_image(),
    scaledown_window=SCALEDOWN_WINDOW,
    keep_warm=KEEP_WARM,
    timeout=600,
    allow_concurrent_inputs=10,
)
class MyNewServer:
    @modal.enter()
    def load_model(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        model_path = f"/models/{MODEL_NAME}"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )
        self.model.eval()

    @modal.asgi_app()
    def serve(self):
        from fastapi import FastAPI, Request
        from pydantic import BaseModel

        app = FastAPI(title="My New Model")

        @app.middleware("http")
        async def auth_middleware(request: Request, call_next):
            if request.url.path in ("/health", "/"):
                return await call_next(request)
            from ai_workers.common.auth import verify_api_key

            await verify_api_key(request)
            return await call_next(request)

        @app.get("/health")
        async def health():
            return {"status": "ok", "model": MODEL_NAME}

        # TODO: Add task-specific endpoints for your model

        return app
```

### Choosing Endpoint Format

| Task       | Endpoint                    | Format           | Reference              |
|------------|-----------------------------|------------------|------------------------|
| Embedding  | `/v1/embeddings`            | OpenAI Embeddings| `workers/embedding.py` |
| Reranker   | `/v1/rerank`                | Cohere Rerank    | `workers/reranker.py`  |
| OCR/Vision | `/v1/chat/completions`      | OpenAI Chat      | `workers/ocr.py`       |
| ASR        | `/v1/audio/transcriptions`  | OpenAI Audio     | `workers/asr.py`       |

### Choosing Modal Image

| Use Case                    | Function                               | Packages                          |
|-----------------------------|----------------------------------------|-----------------------------------|
| Transformers (text/vision)  | `transformers_image()`                 | torch, transformers, Pillow       |
| Transformers + Flash Attn   | `transformers_image(flash_attn=True)`  | + flash-attn                      |
| Audio models                | `transformers_audio_image()`           | + librosa, soundfile              |

## Step 3: Deploy to Modal

```bash
# Dry run (verify before deploying)
mise run deploy my-new-model-7b --dry-run

# Deploy
mise run deploy my-new-model-7b

# Deploy all workers
mise run deploy-all
```

> Mise tasks integrate `infisical run --env=prod --` to inject secrets.

## Step 4: Add to LiteLLM Config

File: `litellm/config.yaml`

```yaml
- model_name: my-new-model-7b
  litellm_params:
    model: openai/my-new-model-7b  # or cohere/ for reranker
    api_base: https://<your-modal-workspace>--ai-workers-my-new-model-7b-mynewserver-serve.modal.run
    api_key: ${WORKER_API_KEY}
```

> Get the exact URL from the Modal dashboard after deployment.

## Step 5: Test Endpoint

```bash
# Test directly (bypassing LiteLLM)
curl -X POST https://<modal-url>/v1/embeddings \
  -H "Authorization: Bearer $WORKER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "my-new-model-7b", "input": ["test"]}'

# Health check
curl https://<modal-url>/health
```

## Checklist

- [ ] Add `ModelConfig` to `config.py`
- [ ] Create worker module in `workers/`
- [ ] Deploy to Modal
- [ ] Add to `litellm/config.yaml`
- [ ] Test endpoint works
- [ ] Update README worker matrix
