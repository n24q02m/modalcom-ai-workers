# Huong dan them Model moi

Tai lieu nay mo ta cac buoc de them mot model AI moi vao he thong workers.

## Tong quan quy trinh

```
1. Them vao Model Registry (config.py)
2. Tao Worker module (workers/*.py)
3. Convert weights (CLI)
4. Upload len R2 (CLI)
5. Deploy len Modal (CLI)
6. Them vao LiteLLM config
7. Test endpoint
```

## Buoc 1: Them vao Model Registry

File: `src/ai_workers/common/config.py`

Them mot entry moi vao cuoi phan Model Registry:

```python
_register(
    ModelConfig(
        # Ten duy nhat, dung lam key va R2 prefix
        name="my-new-model-7b",
        # HuggingFace model ID (dung de download khi convert)
        hf_id="organization/MyNewModel-7B",
        # Task: EMBEDDING, RERANKER_LLM, VL_EMBEDDING, VL_RERANKER, OCR, ASR
        task=Task.EMBEDDING,
        # Tier: LIGHT hoac HEAVY
        tier=Tier.HEAVY,
        # Precision: FP16 (mac dinh) hoac BF16
        # Chi dung BF16 khi model duoc train bang BF16 va khong the convert FP16
        precision=Precision.FP16,
        # GPU: T4 (models <= 2B) hoac A10G (models > 2B hoac BF16)
        gpu=GPU.A10G,
        # Serving: VLLM (chi cho embedding text) hoac CUSTOM_FASTAPI
        serving_engine=ServingEngine.VLLM,
        # Model class: AUTO_MODEL, CAUSAL_LM, SEQ2SEQ
        model_class=ModelClassType.AUTO_MODEL,
        # Trust remote code (True cho Qwen, DeepSeek, etc.)
        trust_remote_code=True,
        # Module path cua worker file
        worker_module="ai_workers.workers.my_new_worker",
        # Extra kwargs cho model loading (optional)
        extra_load_kwargs={},
    )
)
```

### Chon GPU

| Model Size | Precision | GPU | Ly do |
|-----------|-----------|-----|-------|
| <= 2B | FP16 | T4 | Du VRAM (16GB) |
| 3-8B | FP16 | A10G | Can 24GB VRAM |
| Bat ky | BF16 | A10G+ | T4 khong ho tro BF16 |

### Chon Precision

- **FP16** (mac dinh): Phu hop hau het models. Nho hon, nhanh hon.
- **BF16**: Chi khi model duoc train bang BF16 va convert FP16 gay degradation (vd: DeepSeek-OCR-2).

## Buoc 2: Tao Worker Module

File: `src/ai_workers/workers/my_new_worker.py`

### Template co ban

```python
"""My New Worker description."""

from __future__ import annotations

import modal

from ai_workers.common.images import MODELS_MOUNT_PATH, transformers_image
from ai_workers.common.r2 import get_modal_cloud_bucket_mount

SCALEDOWN_WINDOW = 300
KEEP_WARM = 0
MODEL_NAME = "my-new-model-7b"

r2_mount = get_modal_cloud_bucket_mount()

my_app = modal.App(
    "ai-workers-my-new-model-7b",
    secrets=[
        modal.Secret.from_name("r2-credentials"),
        modal.Secret.from_name("worker-api-key"),
    ],
)


@my_app.cls(
    gpu="A10G",
    image=transformers_image(),
    volumes={MODELS_MOUNT_PATH: r2_mount},
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

        model_path = f"{MODELS_MOUNT_PATH}/{MODEL_NAME}"
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

        # --- SAMPLE ENDPOINT: Chat Completion (Text Generation / Vision) ---
        #
        # class ChatCompletionRequest(BaseModel):
        #     model: str
        #     messages: list[dict]
        #     max_tokens: int = 1024
        #
        # @app.post("/v1/chat/completions")
        # async def chat(request: ChatCompletionRequest):
        #     # Implement model inference logic here
        #     # ...
        #     return {
        #         "id": "chatcmpl-123",
        #         "object": "chat.completion",
        #         "created": 1677652288,
        #         "model": request.model,
        #         "choices": [{
        #             "index": 0,
        #             "message": {"role": "assistant", "content": "Hello!"},
        #             "finish_reason": "stop"
        #         }],
        #         "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21}
        #     }
        #
        # Note: Check table below for specific endpoint formats (Embeddings, Reranker, etc.)

        return app
```

### Chon endpoint format

| Task | Endpoint | Format | Reference |
|------|----------|--------|-----------|
| Embedding | `/v1/embeddings` | OpenAI Embeddings | `workers/embedding.py` |
| Reranker | `/v1/rerank` | Cohere Rerank | `workers/reranker.py` |
| OCR / Vision | `/v1/chat/completions` | OpenAI Chat | `workers/ocr.py` |
| ASR | `/v1/audio/transcriptions` | OpenAI Audio | `workers/asr.py` |

### Chon Modal Image

| Use Case | Function | Packages |
|----------|----------|----------|
| vLLM embedding | `vllm_image()` | vllm, torch, transformers |
| Transformers (text/vision) | `transformers_image()` | torch, transformers, Pillow |
| Transformers + Flash Attn | `transformers_image(flash_attn=True)` | + flash-attn |
| Audio models | `transformers_audio_image()` | + librosa, soundfile |

## Buoc 3: Convert Weights

```bash
# Convert model tu HuggingFace sang target precision
python -m ai_workers convert run my-new-model-7b

# Kiem tra output
ls converted/my-new-model-7b/
```

Output se nam trong `./converted/my-new-model-7b/` voi SafeTensors format.

> **Luu y:** Local machine co 4GB VRAM + 16GB RAM.
> - Models <= 8B FP16 (16GB): vua du, can `low_cpu_mem_usage=True`
> - Models > 8B: can convert tren may khac hoac dung `--device cpu` va swap

## Buoc 4: Upload len R2

```bash
# Upload len CF R2
python -m ai_workers upload run my-new-model-7b

# Voi backup GDrive
python -m ai_workers upload run my-new-model-7b --backup-gdrive
```

## Buoc 5: Deploy len Modal

```bash
# Dry run (kiem tra truoc)
python -m ai_workers deploy run my-new-model-7b --dry-run

# Deploy that
python -m ai_workers deploy run my-new-model-7b
```

## Buoc 6: Them vao LiteLLM Config

File: `litellm/config.yaml`

```yaml
- model_name: my-new-model-7b
  litellm_params:
    model: openai/my-new-model-7b  # hoac cohere/ cho reranker
    api_base: https://<workspace>--ai-workers-my-new-model-7b-mynewserver-serve.modal.run
    api_key: ${WORKER_API_KEY}
```

> Lay URL chinh xac tu Modal dashboard sau khi deploy.

## Buoc 7: Test Endpoint

```bash
# Test truc tiep (khong qua LiteLLM)
curl -X POST https://<modal-url>/v1/embeddings \
  -H "Authorization: Bearer $WORKER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "my-new-model-7b", "input": ["test"]}'

# Health check
curl https://<modal-url>/health
```

## Checklist

- [ ] Them `ModelConfig` vao `config.py`
- [ ] Tao worker module trong `workers/`
- [ ] Convert weights thanh cong
- [ ] Upload len R2
- [ ] Deploy len Modal
- [ ] Them vao `litellm/config.yaml`
- [ ] Test endpoint hoat dong
- [ ] Cap nhat README worker matrix
