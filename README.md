# modalcom-ai-workers

GPU-serverless AI workers trên [Modal.com](https://modal.com) cho embedding, reranking, OCR và ASR.

Tất cả endpoints tương thích với [LiteLLM](https://docs.litellm.ai/) proxy — consumers (KnowledgePrism, EchoVault, ...) giao tiếp thông qua OpenAI/Cohere SDK chuẩn.

## Architecture

```
Consumer Apps (KnowledgePrism, EchoVault)
         │
    LiteLLM Proxy (routing + auth)
         │
    ┌────┼────────────────────────────────────┐
    │    Modal.com GPU Serverless              │
    │                                          │
    │  ┌─────────────┐  ┌─────────────┐       │
    │  │ Embedding    │  │ Reranker    │       │
    │  │ Light (T4)   │  │ Light (T4)  │       │
    │  │ Heavy (A10G) │  │ Heavy (A10G)│       │
    │  └─────────────┘  └─────────────┘       │
    │                                          │
    │  ┌─────────────┐  ┌─────────────┐       │
    │  │ VL Embedding │  │ VL Reranker │       │
    │  │ Light (T4)   │  │ Light (T4)  │       │
    │  │ Heavy (A10G) │  │ Heavy (A10G)│       │
    │  └─────────────┘  └─────────────┘       │
    │                                          │
    │  ┌─────────────┐  ┌─────────────┐       │
    │  │ OCR (A10G)   │  │ ASR (T4)    │       │
    │  │ BF16         │  │ FP16        │       │
    │  └─────────────┘  └─────────────┘       │
    │                                          │
    │  CloudBucketMount ←── CF R2 (weights)   │
    └──────────────────────────────────────────┘
```

## Worker Matrix

| Model | Task | Tier | GPU | Precision | Endpoint |
|-------|------|------|-----|-----------|----------|
| Qwen3-Embedding-0.6B | Embedding | Light | T4 | FP16 | `/v1/embeddings` |
| Qwen3-Embedding-8B | Embedding | Heavy | A10G | FP16 | `/v1/embeddings` |
| Qwen3-Reranker-0.6B | Reranker | Light | T4 | FP16 | `/v1/rerank` |
| Qwen3-Reranker-8B | Reranker | Heavy | A10G | FP16 | `/v1/rerank` |
| Qwen3-VL-Embedding-2B | VL Embed | Light | T4 | FP16 | `/v1/embeddings` |
| Qwen3-VL-Embedding-8B | VL Embed | Heavy | A10G | FP16 | `/v1/embeddings` |
| Qwen3-VL-Reranker-2B | VL Rerank | Light | T4 | FP16 | `/v1/rerank` |
| Qwen3-VL-Reranker-8B | VL Rerank | Heavy | A10G | FP16 | `/v1/rerank` |
| DeepSeek-OCR-2 | OCR | Heavy | A10G | BF16 | `/v1/chat/completions` |
| Whisper-Large-v3 | ASR | Heavy | T4 | FP16 | `/v1/audio/transcriptions` |

## Quick Start

### Prerequisites

- Python 3.13 + [mise](https://mise.jdx.dev/)
- Modal account + token
- CF R2 bucket with model weights

### Setup

```bash
# Install dependencies
mise install
mise run setup

# Or manually
uv sync --all-groups
```

### Pipeline: Convert → Upload → Deploy

```bash
# 1. Convert HuggingFace model to target precision (SafeTensors)
mise run convert qwen3-embedding-0.6b

# 2. Upload converted weights to CF R2
mise run upload qwen3-embedding-0.6b

# 3. Deploy to Modal
mise run deploy qwen3-embedding-0.6b

# Deploy all workers at once
mise run deploy-all
```

### CLI Commands

```bash
# List available models
python -m ai_workers convert list
python -m ai_workers upload list
python -m ai_workers deploy list

# Convert specific model
python -m ai_workers convert run qwen3-embedding-0.6b

# Upload with GDrive backup
python -m ai_workers upload run qwen3-embedding-0.6b --backup-gdrive

# Deploy with dry-run
python -m ai_workers deploy run qwen3-embedding-0.6b --dry-run
```

## Project Structure

```
src/ai_workers/
├── common/
│   ├── config.py       # Model registry (single source of truth)
│   ├── r2.py           # CF R2 storage utilities
│   ├── auth.py         # Bearer token middleware
│   ├── images.py       # Modal container images
│   └── logging.py      # Structured logging
├── cli/
│   ├── __main__.py     # CLI entry point
│   ├── convert.py      # HF → SafeTensors conversion
│   ├── upload.py       # Upload to R2 / GDrive
│   └── deploy.py       # Modal deployment
└── workers/
    ├── embedding.py    # Text embedding (vLLM)
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
mise run lint

# Fix lint issues
mise run fix

# Test
mise run test
```

## Environment Variables

### Modal Secrets

Tao trên Modal dashboard hoặc CLI:

```bash
# R2 credentials (for CloudBucketMount)
modal secret create r2-credentials \
  R2_ENDPOINT_URL="https://xxx.r2.cloudflarestorage.com" \
  R2_ACCESS_KEY_ID="xxx" \
  R2_SECRET_ACCESS_KEY="xxx" \
  R2_BUCKET_NAME="ai-models"

# Worker API key
modal secret create worker-api-key \
  WORKER_API_KEY="your-secret-key"
```

### Local (.env)

```bash
# R2 (for CLI upload)
R2_ENDPOINT_URL=https://xxx.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=xxx
R2_SECRET_ACCESS_KEY=xxx
R2_BUCKET_NAME=ai-models

# Modal (for CLI deploy)
MODAL_TOKEN_ID=xxx
MODAL_TOKEN_SECRET=xxx
```

## Adding New Models

Xem [docs/ADD_NEW_MODEL.md](docs/ADD_NEW_MODEL.md) de biet cach them model moi vào hệ thống.

## License

MIT
