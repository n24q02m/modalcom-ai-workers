# modalcom-ai-workers

**GPU-serverless AI workers on Modal.com for embedding, reranking, OCR, and ASR**

[![CI](https://github.com/n24q02m/modalcom-ai-workers/actions/workflows/ci.yml/badge.svg)](https://github.com/n24q02m/modalcom-ai-workers/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/n24q02m/modalcom-ai-workers/graph/badge.svg?token=5Z9ETF0G7B)](https://codecov.io/gh/n24q02m/modalcom-ai-workers)
[![License](https://img.shields.io/github/license/n24q02m/modalcom-ai-workers)](LICENSE)

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](#)
[![Modal](https://img.shields.io/badge/Modal-000000?logo=modal&logoColor=white)](https://modal.com)
[![LiteLLM](https://img.shields.io/badge/LiteLLM-1A1F6C?logo=openai&logoColor=white)](https://docs.litellm.ai/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![semantic-release](https://img.shields.io/badge/semantic--release-e10079?logo=semantic-release&logoColor=white)](https://github.com/python-semantic-release/python-semantic-release)
[![Renovate](https://img.shields.io/badge/renovate-enabled-1A1F6C?logo=renovatebot&logoColor=white)](https://developer.mend.io/)

All endpoints are [LiteLLM](https://docs.litellm.ai/)-compatible — consumer apps communicate through standard OpenAI/Cohere SDKs.

## Architecture

```
Consumer Apps (OpenAI/Cohere SDK)
       │
  LiteLLM Proxy (routing + auth + cost tracking)
       │
  ┌────┼────────────────────────────────────────┐
  │    Modal.com GPU Serverless                  │
  │                                              │
  │  ┌──────────────┐  ┌──────────────┐         │
  │  │ Embedding    │  │ Reranker     │         │
  │  │ 0.6B + 8B   │  │ 0.6B + 8B   │         │
  │  │ (A10G)      │  │ (A10G)       │         │
  │  └──────────────┘  └──────────────┘         │
  │                                              │
  │  ┌──────────────┐  ┌──────────────┐         │
  │  │ VL Embedding │  │ VL Reranker  │         │
  │  │ 2B + 8B     │  │ 2B + 8B     │         │
  │  │ (A10G)      │  │ (A10G)       │         │
  │  └──────────────┘  └──────────────┘         │
  │                                              │
  │  ┌──────────────┐  ┌──────────────┐         │
  │  │ OCR          │  │ ASR          │         │
  │  │ DeepSeek-2   │  │ Whisper v3   │         │
  │  │ BF16 (A10G) │  │ FP16 (T4)    │         │
  │  └──────────────┘  └──────────────┘         │
  │                                              │
  │  Models loaded from HuggingFace Hub          │
  │  via Xet protocol (~1GB/s)                   │
  └──────────────────────────────────────────────┘
```

Light + Heavy model variants are merged into single Modal apps — both sizes share the same endpoint. Routing is done via the `model` field in the request body. All workers scale to zero when idle (5 min cooldown).

## Worker Matrix

| Model | HuggingFace ID | Task | GPU | Precision | Endpoint |
|-------|---------------|------|-----|-----------|----------|
| `qwen3-embedding-0.6b` | `Qwen/Qwen3-Embedding-0.6B` | Embedding | A10G | FP16 | `/v1/embeddings` |
| `qwen3-embedding-8b` | `Qwen/Qwen3-Embedding-8B` | Embedding | A10G | FP16 | `/v1/embeddings` |
| `qwen3-reranker-0.6b` | `Qwen/Qwen3-Reranker-0.6B` | Reranker | A10G | FP16 | `/v1/rerank` |
| `qwen3-reranker-8b` | `Qwen/Qwen3-Reranker-8B` | Reranker | A10G | FP16 | `/v1/rerank` |
| `qwen3-vl-embedding-2b` | `Qwen/Qwen3-VL-Embedding-2B` | VL Embed | A10G | FP16 | `/v1/embeddings` |
| `qwen3-vl-embedding-8b` | `Qwen/Qwen3-VL-Embedding-8B` | VL Embed | A10G | FP16 | `/v1/embeddings` |
| `qwen3-vl-reranker-2b` | `Qwen/Qwen3-VL-Reranker-2B` | VL Rerank | A10G | FP16 | `/v1/rerank` |
| `qwen3-vl-reranker-8b` | `Qwen/Qwen3-VL-Reranker-8B` | VL Rerank | A10G | FP16 | `/v1/rerank` |
| `deepseek-ocr-2` | `deepseek-ai/DeepSeek-OCR-2` | OCR | A10G | BF16 | `/v1/chat/completions` |
| `whisper-large-v3` | `openai/whisper-large-v3` | ASR | T4 | FP16 | `/v1/audio/transcriptions` |

## Quick Start

### Prerequisites

- Python 3.13+ and [uv](https://docs.astral.sh/uv/)
- [Modal](https://modal.com) account with API token (`modal token new`)
- [HuggingFace](https://huggingface.co) token (for ONNX/GGUF conversion only)

### Setup

```bash
uv sync --all-groups

# Or with mise
mise install && mise run setup
```

### Deploy Workers

```bash
# List available models
python -m ai_workers deploy list

# Deploy a single worker
python -m ai_workers deploy qwen3-embedding-0.6b

# Deploy all workers
python -m ai_workers deploy --all

# Dry run (show what would be deployed)
python -m ai_workers deploy qwen3-embedding-0.6b --dry-run
```

### ONNX/GGUF Conversion

Convert models to optimized formats and push to HuggingFace Hub. Conversion runs on Modal CPU containers (no local GPU needed).

```bash
# ONNX (FP32 → INT8 + Q4F16 quantization)
python -m ai_workers onnx-convert qwen3-embedding-0.6b
python -m ai_workers onnx-convert --all

# GGUF (Q4_K_M via llama.cpp)
python -m ai_workers gguf-convert qwen3-embedding-0.6b
python -m ai_workers gguf-convert --all

# List convertible models
python -m ai_workers onnx-convert list
python -m ai_workers gguf-convert list
```

## LiteLLM Proxy Integration

Workers expose OpenAI/Cohere-compatible endpoints. Use [LiteLLM Proxy](https://docs.litellm.ai/) for unified routing, auth, and cost tracking.

A ready-to-use proxy config is provided at [`litellm/config.yaml`](litellm/config.yaml). Replace `<your-modal-workspace>` with your Modal workspace name.

### Model Naming Convention

| Task | LiteLLM Prefix | Example |
|------|---------------|---------|
| Embedding | `openai/` | `openai/qwen3-embedding-0.6b` |
| Reranker | `cohere/` | `cohere/qwen3-reranker-0.6b` |
| VL Embedding | `openai/` | `openai/qwen3-vl-embedding-2b` |
| VL Reranker | `cohere/` | `cohere/qwen3-vl-reranker-2b` |
| OCR | `openai/` | `openai/deepseek-ocr-2` |
| ASR | `openai/` | `openai/whisper-large-v3` |

### Proxy Config Example

```yaml
model_list:
  - model_name: qwen3-embedding-0.6b
    litellm_params:
      model: openai/qwen3-embedding-0.6b
      api_base: https://<workspace>--ai-workers-embedding-embeddingserver-serve.modal.run
      api_key: ${WORKER_API_KEY}
```

### Consumer Usage (Python)

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

# OCR (OpenAI Vision-compatible)
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

### Consumer Usage (curl)

```bash
# Embedding
curl -X POST http://localhost:4000/v1/embeddings \
  -H "Authorization: Bearer your-litellm-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-embedding-0.6b", "input": ["Hello"]}'

# Rerank (Cohere-compatible)
curl -X POST http://localhost:4000/v1/rerank \
  -H "Authorization: Bearer your-litellm-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-reranker-0.6b", "query": "What is AI?", "documents": ["AI is...", "Cats are..."]}'
```

### Modal Endpoint URL Format

```
https://<workspace>--<app-name>-<classname-lowercase>-serve.modal.run
```

After deployment, get the exact URL from `modal app list` or the Modal dashboard.

## Project Structure

```
src/ai_workers/
├── common/
│   ├── config.py       # Model registry (single source of truth)
│   ├── auth.py         # Bearer token middleware
│   ├── images.py       # Modal container images
│   └── logging.py      # Structured logging
├── cli/
│   ├── __main__.py     # CLI entry point (deploy, onnx-convert, gguf-convert)
│   ├── deploy.py       # Deploy workers to Modal
│   ├── onnx_convert.py # ONNX conversion CLI
│   └── gguf_convert.py # GGUF conversion CLI
└── workers/
    ├── embedding.py    # Text embedding (Qwen3-Embedding 0.6B + 8B)
    ├── reranker.py     # Text reranker (Qwen3-Reranker 0.6B + 8B)
    ├── vl_embedding.py # Vision-Language embedding (Qwen3-VL-Embedding 2B + 8B)
    ├── vl_reranker.py  # Vision-Language reranker (Qwen3-VL-Reranker 2B + 8B)
    ├── ocr.py          # Document OCR (DeepSeek-OCR-2)
    ├── asr.py          # Speech-to-text (Whisper-Large-v3)
    ├── onnx_converter.py  # ONNX export + quantization (Modal CPU)
    └── gguf_converter.py  # GGUF conversion via llama.cpp (Modal CPU)
litellm/
├── config.yaml         # LiteLLM proxy routing config template
└── README.md           # Proxy setup guide
```

## Adding a New Model

All model metadata lives in `src/ai_workers/common/config.py`. To add a new model:

1. **Register in config.py** — add a `_register(ModelConfig(...))` call:

   ```python
   _register(
       ModelConfig(
           name="my-new-model",              # Registry key
           hf_id="org/MyNewModel",           # HuggingFace model ID
           task=Task.EMBEDDING,              # Task type
           tier=Tier.LIGHT,                  # LIGHT or HEAVY
           precision=Precision.FP16,         # FP16 or BF16
           gpu=GPU.A10G,                     # A10G or T4
           model_class=ModelClassType.AUTO_MODEL,
           worker_module="ai_workers.workers.embedding",
           modal_app_var="embedding_app",
           modal_app_name="ai-workers-embedding",
       )
   )
   ```

2. **Create or reuse a worker** — if the model fits an existing worker pattern (embedding, reranker, etc.), point `worker_module` to that file. For a new task type, create a new worker file in `workers/`.

3. **Add LiteLLM config** — add the model to `litellm/config.yaml`.

4. **Deploy** — `python -m ai_workers deploy my-new-model`.

## Secrets

### Modal Secrets (created on Modal dashboard)

| Secret Name | Key | Used By |
|------------|-----|---------|
| `worker-api-key` | `WORKER_API_KEY` | All serving workers (endpoint auth) |
| `hf-token` | `HF_TOKEN` | ONNX/GGUF converters (HuggingFace Hub push) |

### CI/CD Secrets (GitHub Actions)

| Secret | Description |
|--------|-------------|
| `MODAL_TOKEN_ID` | Modal API token ID |
| `MODAL_TOKEN_SECRET` | Modal API token secret |
| `HF_TOKEN` | HuggingFace token (for conversion jobs) |
| `GH_PAT` | GitHub PAT (for semantic-release) |

## Development

```bash
# Lint + format check
uv run ruff check . && uv run ruff format --check .

# Type check
uv run ty check

# Test
uv run pytest

# Auto-fix lint issues
uv run ruff check --fix . && uv run ruff format .
```

### Commit Convention

This project uses [Conventional Commits](https://www.conventionalcommits.org/). A pre-commit hook enforces the format:

```
<type>(<optional scope>): <description>
```

Allowed types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.

## Related Projects

- [qwen3-embed](https://github.com/n24q02m/qwen3-embed) — Local ONNX runtime for Qwen3 embedding models (offline/edge deployment)

## License

[MIT](LICENSE)
