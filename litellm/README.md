# LiteLLM Proxy Configuration

The `config.yaml` file configures LiteLLM Proxy to route requests to Modal workers.

## Setup

1. Replace `<your-modal-workspace>` with your actual Modal workspace name
2. Set environment variables:
   - `LITELLM_MASTER_KEY`: API key for the LiteLLM proxy
   - `WORKER_API_KEY`: Shared API key for all Modal workers

3. Run the proxy:
   ```bash
   litellm --config litellm/config.yaml --port 4000
   ```

## Model Naming Convention

| Task         | LiteLLM Prefix | Example                          |
|--------------|---------------|----------------------------------|
| Embedding    | `openai/`     | `openai/qwen3-embedding-0.6b`   |
| Reranker     | `cohere/`     | `cohere/qwen3-reranker-0.6b`    |
| VL Embedding | `openai/`     | `openai/qwen3-vl-embedding-2b`  |
| VL Reranker  | `cohere/`     | `cohere/qwen3-vl-reranker-2b`   |
| OCR          | `openai/`     | `openai/deepseek-ocr-2`         |
| ASR          | `openai/`     | `openai/whisper-large-v3`        |

## Consumer Usage

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

## Endpoint URL Format

Modal endpoint URL format:
```
https://<workspace>--<app-name>-<classname-lowercase>-serve.modal.run
```

Example: workspace `my-workspace`, app `ai-workers-embedding`, class `EmbeddingServer`:
```
https://my-workspace--ai-workers-embedding-embeddingserver-serve.modal.run
```

> **Note:** After the first deployment, get the exact URL from `modal app list` or the Modal dashboard.
