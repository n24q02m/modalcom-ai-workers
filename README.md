# modalcom-ai-workers

[![CI](https://github.com/n24q02m/modalcom-ai-workers/actions/workflows/ci.yml/badge.svg)](https://github.com/n24q02m/modalcom-ai-workers/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)

GPU-serverless AI workers on [Modal.com](https://modal.com) for embedding, reranking, OCR, and ASR.

All endpoints are [LiteLLM](https://docs.litellm.ai/)-compatible — consumer apps communicate through standard OpenAI/Cohere SDKs.

## Architecture

```
Consumer Apps
      │
 LiteLLM Proxy (routing + auth)
      │
 ┌────┼────────────────────────────────────┐
 │    Modal.com GPU Serverless              │
 │                                          │
 │  ┌─────────────┐  ┌─────────────┐       │
 │  │ Embedding    │  │ Reranker    │       │
 │  │ (A10G)      │  │ (A10G)      │       │
 │  └─────────────┘  └─────────────┘       │
 │                                          │
 │  ┌─────────────┐  ┌─────────────┐       │
 │  │ VL Embedding │  │ VL Reranker │       │
 │  │ (A10G)      │  │ (A10G)      │       │
 │  └─────────────┘  └─────────────┘       │
 │                                          │
 │  ┌─────────────┐  ┌─────────────┐       │
 │  │ OCR (A10G)   │  │ ASR (T4)    │       │
 │  │ BF16         │  │ FP16        │       │
 │  └─────────────┘  └─────────────┘       │
 │                                          │
 │  Models loaded from HuggingFace Hub      │
 │  via Xet protocol (~1GB/s)               │
 └──────────────────────────────────────────┘
```

## Worker Matrix

| Model | Task | GPU | Precision | Endpoint |
|-------|------|-----|-----------|----------|
| Qwen3-Embedding-0.6B | Embedding | A10G | FP16 | `/v1/embeddings` |
| Qwen3-Embedding-8B | Embedding | A10G | FP16 | `/v1/embeddings` |
| Qwen3-Reranker-0.6B | Reranker | A10G | FP16 | `/v1/rerank` |
| Qwen3-Reranker-8B | Reranker | A10G | FP16 | `/v1/rerank` |
| Qwen3-VL-Embedding-2B | VL Embed | A10G | FP16 | `/v1/embeddings` |
| Qwen3-VL-Embedding-8B | VL Embed | A10G | FP16 | `/v1/embeddings` |
| Qwen3-VL-Reranker-2B | VL Rerank | A10G | FP16 | `/v1/rerank` |
| Qwen3-VL-Reranker-8B | VL Rerank | A10G | FP16 | `/v1/rerank` |
| DeepSeek-OCR-2 | OCR | A10G | BF16 | `/v1/chat/completions` |
| Whisper-Large-v3 | ASR | T4 | FP16 | `/v1/audio/transcriptions` |

## Quick Start

### Prerequisites

- Python 3.13 + [uv](https://docs.astral.sh/uv/)
- [Modal](https://modal.com) account + token (`modal token new`)

### Setup

```bash
# Install dependencies
uv sync --all-groups

# Or with mise
mise install && mise run setup
```

### Deploy

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

Convert models to optimized formats and push to HuggingFace Hub:

```bash
# ONNX (INT8 + Q4F16)
python -m ai_workers onnx-convert qwen3-embedding-0.6b

# GGUF (Q4_K_M via llama.cpp)
python -m ai_workers gguf-convert qwen3-embedding-0.6b
```

## LiteLLM Integration

Workers expose OpenAI/Cohere-compatible endpoints. See [litellm/README.md](litellm/README.md) for proxy configuration and [litellm/config.yaml](litellm/config.yaml) for a ready-to-use config template.

```yaml
# Example: register embedding worker in LiteLLM proxy
model_list:
  - model_name: qwen3-embedding-0.6b
    litellm_params:
      model: openai/qwen3-embedding-0.6b
      api_base: https://<workspace>--ai-workers-embedding-serve.modal.run
      api_key: your-worker-api-key
```

## Project Structure

```
src/ai_workers/
├── common/
│   ├── config.py       # Model registry (single source of truth)
│   ├── auth.py         # Bearer token middleware
│   ├── images.py       # Modal container images
│   └── logging.py      # Structured logging
├── cli/
│   ├── __main__.py     # CLI entry point
│   ├── deploy.py       # Deploy workers to Modal
│   ├── onnx_convert.py # ONNX conversion (INT8 + Q4F16)
│   └── gguf_convert.py # GGUF conversion (Q4_K_M)
└── workers/
    ├── embedding.py    # Text embedding
    ├── reranker.py     # Text reranker (yes/no scoring)
    ├── vl_embedding.py # Vision-Language embedding
    ├── vl_reranker.py  # Vision-Language reranker
    ├── ocr.py          # Document OCR (DeepSeek)
    └── asr.py          # Speech-to-text (Whisper)
litellm/
├── config.yaml         # LiteLLM proxy routing config
└── README.md           # Consumer integration guide
docs/
└── ADD_NEW_MODEL.md    # Guide to add new models
```

## Development

```bash
# Lint
uv run ruff check . && uv run ruff format --check .

# Type check
uv run ty check

# Test
uv run pytest

# Auto-fix
uv run ruff check --fix . && uv run ruff format .
```

## Secrets

### Environment Variables (for deployment)

| Variable | Description | Used by |
|----------|-------------|---------|
| `MODAL_TOKEN_ID` | Modal API token ID | Deploy |
| `MODAL_TOKEN_SECRET` | Modal API token secret | Deploy |

### Modal Secrets (on Modal dashboard)

```bash
# Worker API key (protects endpoints)
modal secret create worker-api-key \
  WORKER_API_KEY="your-secret-key"
```

## Adding New Models

See [docs/ADD_NEW_MODEL.md](docs/ADD_NEW_MODEL.md) for a step-by-step guide.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

[MIT](LICENSE)
