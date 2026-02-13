# Hướng dẫn thêm Model mới

Tài liệu này mô tả các bước để thêm một model AI mới vào hệ thống workers.

## Tổng quan quy trình

```
1. Thêm vào Model Registry (config.py)
2. Tạo Worker module (workers/*.py)
3. Convert weights (chạy trên Modal CPU → ghi thẳng R2)
4. Deploy lên Modal (CLI)
5. Thêm vào LiteLLM config
6. Test endpoint
```

## Bước 1: Thêm vào Model Registry

File: `src/ai_workers/common/config.py`

Thêm một entry mới vào cuối phần Model Registry:

```python
_register(
    ModelConfig(
        # Tên duy nhất, dùng làm key và R2 prefix
        name="my-new-model-7b",
        # HuggingFace model ID (dùng để download khi convert)
        hf_id="organization/MyNewModel-7B",
        # Task: EMBEDDING, RERANKER_LLM, VL_EMBEDDING, VL_RERANKER, OCR, ASR
        task=Task.EMBEDDING,
        # Tier: LIGHT hoặc HEAVY
        tier=Tier.HEAVY,
        # Precision: FP16 (mặc định) hoặc BF16
        # Chỉ dùng BF16 khi model được train bằng BF16 và không thể convert sang FP16
        precision=Precision.FP16,
        # GPU: T4 (models <= 2B) hoặc A10G (models > 2B hoặc BF16)
        gpu=GPU.A10G,
        # Serving: VLLM (chỉ cho embedding text) hoặc CUSTOM_FASTAPI
        serving_engine=ServingEngine.VLLM,
        # Model class: AUTO_MODEL, CAUSAL_LM, SEQ2SEQ
        model_class=ModelClassType.AUTO_MODEL,
        # Trust remote code (True cho Qwen, DeepSeek, ...)
        trust_remote_code=True,
        # Module path của worker file
        worker_module="ai_workers.workers.my_new_worker",
        # Extra kwargs cho model loading (tuỳ chọn)
        extra_load_kwargs={},
    )
)
```

### Chọn GPU

| Kích thước Model | Precision | GPU | Lý do |
|-----------|-----------|-----|-------|
| <= 2B | FP16 | T4 | Đủ VRAM (16GB) |
| 3-8B | FP16 | A10G | Cần 24GB VRAM |
| Bất kỳ | BF16 | A10G+ | T4 không hỗ trợ BF16 |

### Chọn Precision

- **FP16** (mặc định): Phù hợp hầu hết models. Nhỏ hơn, nhanh hơn.
- **BF16**: Chỉ khi model được train bằng BF16 và convert sang FP16 gây suy giảm chất lượng (ví dụ: DeepSeek-OCR-2).

## Bước 2: Tạo Worker Module

File: `src/ai_workers/workers/my_new_worker.py`

### Template cơ bản

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

        # TODO: Thêm các endpoint cụ thể cho task của model

        return app
```

### Chọn endpoint format

| Task | Endpoint | Format | Reference |
|------|----------|--------|-----------|
| Embedding | `/v1/embeddings` | OpenAI Embeddings | `workers/embedding.py` |
| Reranker | `/v1/rerank` | Cohere Rerank | `workers/reranker.py` |
| OCR / Vision | `/v1/chat/completions` | OpenAI Chat | `workers/ocr.py` |
| ASR | `/v1/audio/transcriptions` | OpenAI Audio | `workers/asr.py` |

### Chọn Modal Image

| Use Case | Function | Packages |
|----------|----------|----------|
| vLLM embedding | `vllm_image()` | vllm, torch, transformers |
| Transformers (text/vision) | `transformers_image()` | torch, transformers, Pillow |
| Transformers + Flash Attn | `transformers_image(flash_attn=True)` | + flash-attn |
| Audio models | `transformers_audio_image()` | + librosa, soundfile |

## Bước 3: Convert Weights

Convert chạy trên Modal CPU container (32GB RAM, 4 CPU cores) và ghi thẳng
weights lên R2 qua CloudBucketMount (writable). Không cần máy local mạnh.

```bash
# Convert model — chạy trên Modal CPU, output ghi thẳng lên R2
mise run convert my-new-model-7b

# Ghi đè nếu đã tồn tại trên R2
mise run convert my-new-model-7b --force

# Convert tất cả models
mise run convert all
```

> mise tasks đã tích hợp `infisical run --env=prod --` để inject secrets.
> Cần cấu hình Modal secret `r2-credentials` trên Modal dashboard (quyền write + list).

## Bước 4: Deploy lên Modal

```bash
# Dry run (kiểm tra trước)
mise run deploy my-new-model-7b --dry-run

# Deploy thật
mise run deploy my-new-model-7b

# Deploy tất cả workers
mise run deploy-all
```

> mise tasks đã tích hợp `infisical run --env=prod --` để inject secrets.

## Bước 5: Thêm vào LiteLLM Config

File: `litellm/config.yaml`

```yaml
- model_name: my-new-model-7b
  litellm_params:
    model: openai/my-new-model-7b  # hoặc cohere/ cho reranker
    api_base: https://<workspace>--ai-workers-my-new-model-7b-mynewserver-serve.modal.run
    api_key: ${WORKER_API_KEY}
```

> Lấy URL chính xác từ Modal dashboard sau khi deploy.

## Bước 6: Test Endpoint

```bash
# Test trực tiếp (không qua LiteLLM)
curl -X POST https://<modal-url>/v1/embeddings \
  -H "Authorization: Bearer $WORKER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "my-new-model-7b", "input": ["test"]}'

# Health check
curl https://<modal-url>/health
```

## Danh sách kiểm tra

- [ ] Thêm `ModelConfig` vào `config.py`
- [ ] Tạo worker module trong `workers/`
- [ ] Convert weights (Modal CPU → R2)
- [ ] Deploy lên Modal
- [ ] Thêm vào `litellm/config.yaml`
- [ ] Test endpoint hoạt động
- [ ] Cập nhật README worker matrix
