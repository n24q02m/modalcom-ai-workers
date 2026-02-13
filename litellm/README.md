# Cấu hình LiteLLM Proxy

File `config.yaml` cấu hình LiteLLM Proxy để route các request đến các Modal workers.

## Cài đặt

1. Thay `<your-modal-workspace>` bằng tên workspace Modal thực tế
2. Thiết lập biến môi trường:
   - `LITELLM_MASTER_KEY`: API key cho LiteLLM proxy
   - `WORKER_API_KEY`: API key chung cho tất cả Modal workers

3. Chạy proxy:
   ```bash
   litellm --config litellm/config.yaml --port 4000
   ```

## Quy ước đặt tên Model

| Task | LiteLLM Prefix | Ví dụ |
|------|---------------|-------|
| Embedding | `openai/` | `openai/qwen3-embedding-0.6b` |
| Reranker | `cohere/` | `cohere/qwen3-reranker-0.6b` |
| VL Embedding | `openai/` | `openai/qwen3-vl-embedding-2b` |
| VL Reranker | `cohere/` | `cohere/qwen3-vl-reranker-2b` |
| OCR | `openai/` | `openai/deepseek-ocr-2` |
| ASR | `openai/` | `openai/whisper-large-v3` |

## Sử dụng từ phía Consumer

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-litellm-key",
    base_url="http://localhost:4000",
)

# Embedding
response = client.embeddings.create(
    model="qwen3-embedding-0.6b",
    input=["Hello world"],
)

# OCR
response = client.chat.completions.create(
    model="deepseek-ocr-2",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Extract all text from this image"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]
    }],
)

# ASR
with open("audio.mp3", "rb") as f:
    response = client.audio.transcriptions.create(
        model="whisper-large-v3",
        file=f,
    )
```

### curl

```bash
# Embedding
curl -X POST http://localhost:4000/v1/embeddings \
  -H "Authorization: Bearer your-litellm-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-embedding-0.6b", "input": ["Hello"]}'

# Rerank
curl -X POST http://localhost:4000/v1/rerank \
  -H "Authorization: Bearer your-litellm-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-reranker-0.6b", "query": "What is AI?", "documents": ["AI is...", "Cats are..."]}'
```

## Định dạng URL Endpoint

Định dạng URL endpoint của Modal:
```
https://<workspace>--<app-name>-<classname-lowercase>-serve.modal.run
```

Ví dụ: workspace `my-workspace`, app `ai-workers-qwen3-embedding-0-6b`, class `EmbeddingLightServer`:
```
https://my-workspace--ai-workers-qwen3-embedding-0-6b-embeddinglightserver-serve.modal.run
```

> **Lưu ý:** Sau khi deploy lần đầu, lấy URL chính xác bằng `modal app list` hoặc Modal dashboard.
