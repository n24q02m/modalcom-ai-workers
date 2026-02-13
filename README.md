# modalcom-ai-workers

GPU-serverless AI workers trên [Modal.com](https://modal.com) cho embedding, reranking, OCR và ASR.

Tất cả endpoints tương thích với [LiteLLM](https://docs.litellm.ai/) proxy — các ứng dụng tiêu thụ (KnowledgePrism, EchoVault, ...) giao tiếp thông qua OpenAI/Cohere SDK chuẩn.

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
    │  ┌──────────────────────────────┐       │
    │  │ Converter (CPU, 32GB RAM)    │       │
    │  │ HF Hub → SafeTensors → R2   │       │
    │  └──────────────────────────────┘       │
    │                                          │
    │  CloudBucketMount ←→ CF R2 (weights)    │
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
- [Infisical CLI](https://infisical.com/docs/cli/overview) (quản lý secrets)
- Modal account + token
- CF R2 bucket for model weights

### Setup

```bash
# Install dependencies
mise install
mise run setup

# Or manually
uv sync --all-groups
```

### Pipeline: Convert → Deploy

Convert chạy trên Modal CPU (32GB RAM) và ghi thẳng weights lên R2
qua CloudBucketMount — không cần máy local mạnh, không cần upload riêng.

```bash
# 1. Convert HF model → SafeTensors → ghi thẳng lên R2 (chạy trên Modal CPU)
mise run convert qwen3-embedding-0.6b

# 2. Deploy worker lên Modal (GPU serverless)
mise run deploy qwen3-embedding-0.6b

# Deploy tất cả workers
mise run deploy-all
```

### CLI Commands

```bash
# Liệt kê models trong registry
python -m ai_workers convert list
python -m ai_workers deploy list

# Convert model (chạy trên Modal CPU, ghi thẳng R2)
python -m ai_workers convert qwen3-embedding-0.6b
python -m ai_workers convert all              # convert tất cả
python -m ai_workers convert qwen3-embedding-0.6b --force  # ghi đè

# Upload thủ công (chỉ khi cần upload từ local)
python -m ai_workers upload qwen3-embedding-0.6b --backup-gdrive

# Deploy lên Modal
python -m ai_workers deploy qwen3-embedding-0.6b --dry-run
python -m ai_workers deploy --all
```

## Project Structure

```
src/ai_workers/
├── common/
│   ├── config.py       # Model registry (single source of truth)
│   ├── r2.py           # CF R2 storage + CloudBucketMount
│   ├── auth.py         # Bearer token middleware
│   ├── images.py       # Modal container images (workers + converter)
│   └── logging.py      # Structured logging
├── cli/
│   ├── __main__.py     # CLI entry point
│   ├── convert.py      # Convert qua Modal CPU → ghi thẳng R2
│   ├── upload.py       # Upload thủ công từ local → R2 (tuỳ chọn)
│   └── deploy.py       # Deploy workers lên Modal
└── workers/
    ├── converter.py    # Modal CPU app — convert HF → SafeTensors → R2
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

## CI (GitHub Actions)

| Workflow | File | Trigger | Mô tả |
|----------|------|---------|-------|
| CI | `ci.yml` | PR + push → main | Lint, type check, test |

Convert và deploy chạy CLI local → trigger Modal.com. Xem [Pipeline: Convert → Deploy](#pipeline-convert--deploy).

## Secrets

### Infisical (App Secrets)

6 secrets trong Infisical (env=prod), inject qua `infisical run --env=prod --`:

| Secret | Mô tả | Dùng bởi |
|--------|-------|----------|
| `MODAL_TOKEN_ID` | Modal API token ID | Convert, Deploy |
| `MODAL_TOKEN_SECRET` | Modal API token secret | Convert, Deploy |
| `R2_ENDPOINT_URL` | R2 endpoint (`https://<account-id>.r2.cloudflarestorage.com`) | Convert, Deploy |
| `R2_ACCESS_KEY_ID` | R2 API token access key | Upload (boto3) |
| `R2_SECRET_ACCESS_KEY` | R2 API token secret key | Upload (boto3) |
| `R2_BUCKET_NAME` | R2 bucket name | Convert, Deploy, Upload |

> **R2_ENDPOINT_URL** và **R2_BUCKET_NAME** cần có khi chạy `modal deploy` hoặc convert.
> CloudBucketMount được resolve lúc import module (module-level `get_modal_cloud_bucket_mount()`
> đọc env vars), KHÔNG phải lúc container chạy.

mise tasks (`convert`, `deploy`, `upload`, `deploy-all`) đã tích hợp sẵn `infisical run --env=prod --`.

### Modal Secrets (trên Modal dashboard)

Tạo trên Modal dashboard hoặc CLI:

```bash
# R2 credentials (cho CloudBucketMount — bắt buộc dùng tên key S3-compatible)
modal secret create r2-credentials \
  AWS_ACCESS_KEY_ID="<r2-access-key-id>" \
  AWS_SECRET_ACCESS_KEY="<r2-secret-access-key>"

# Worker API key
modal secret create worker-api-key \
  WORKER_API_KEY="your-secret-key"
```

> **Lưu ý:** CloudBucketMount yêu cầu secret có key names `AWS_ACCESS_KEY_ID` và
> `AWS_SECRET_ACCESS_KEY` (S3-compatible). Đây là giá trị của R2 API token,
> KHÔNG phải AWS credentials.
>
> R2 token cần quyền **read + write + list** (cho converter ghi weights lên R2).
> Workers chỉ đọc (read-only mount), converter ghi (writable mount).

## Thêm Model Mới

Xem [docs/ADD_NEW_MODEL.md](docs/ADD_NEW_MODEL.md) để biết cách thêm model mới vào hệ thống.

## License

MIT
