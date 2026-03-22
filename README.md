# modalcom-ai-workers

**GPU-serverless AI workers on Modal.com for embedding, reranking, OCR, TTS, and ASR**

<!-- Badge Row 1: Status -->
[![CI](https://github.com/n24q02m/modalcom-ai-workers/actions/workflows/ci.yml/badge.svg)](https://github.com/n24q02m/modalcom-ai-workers/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/n24q02m/modalcom-ai-workers/graph/badge.svg?token=5Z9ETF0G7B)](https://codecov.io/gh/n24q02m/modalcom-ai-workers)
[![License](https://img.shields.io/github/license/n24q02m/modalcom-ai-workers)](LICENSE)

<!-- Badge Row 2: Tech -->
[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](#)
[![Modal](https://img.shields.io/badge/Modal-000000?logo=modal&logoColor=white)](https://modal.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LiteLLM](https://img.shields.io/badge/LiteLLM-1A1F6C?logo=openai&logoColor=white)](https://docs.litellm.ai/)
[![semantic-release](https://img.shields.io/badge/semantic--release-e10079?logo=semantic-release&logoColor=white)](https://github.com/python-semantic-release/python-semantic-release)
[![Renovate](https://img.shields.io/badge/renovate-enabled-1A1F6C?logo=renovatebot&logoColor=white)](https://developer.mend.io/)

All endpoints are [LiteLLM](https://docs.litellm.ai/)-compatible -- consumer apps communicate through standard OpenAI/Cohere SDKs. All workers scale to zero when idle (5 min cooldown).

## Features

- **Scale-to-zero GPU**: All workers run on A10G GPUs with automatic 5-minute idle shutdown
- **LiteLLM-compatible**: Standard OpenAI/Cohere SDK endpoints for seamless integration
- **Per-app auth**: Isolated API keys per consumer app (e.g., `KLPRISM_WORKER_API_KEY`, `AIORA_WORKER_API_KEY`)
- **ONNX/GGUF conversion**: Convert models to optimized formats on Modal CPU containers (no local GPU needed)
- **CLI management**: Deploy, list, and convert models via `python -m ai_workers`

## Worker Matrix

### Currently Deployed

| Model | HuggingFace ID | Task | GPU | Precision | Endpoint |
|:------|:---------------|:-----|:----|:----------|:---------|
| `qwen3-reranker-8b` | `Qwen/Qwen3-Reranker-8B` | Reranker | A10G | FP16 | `/v1/rerank` |
| `qwen3-vl-reranker-8b` | `Qwen/Qwen3-VL-Reranker-8B` | VL Rerank | A10G | FP16 | `/v1/rerank` |

### Available (Not Deployed)

| Model | HuggingFace ID | Task | GPU | Precision | Endpoint |
|:------|:---------------|:-----|:----|:----------|:---------|
| `qwen3-embedding-0.6b` | `Qwen/Qwen3-Embedding-0.6B` | Embedding | A10G | FP16 | `/v1/embeddings` |
| `qwen3-embedding-8b` | `Qwen/Qwen3-Embedding-8B` | Embedding | A10G | FP16 | `/v1/embeddings` |
| `qwen3-vl-embedding-2b` | `Qwen/Qwen3-VL-Embedding-2B` | VL Embed | A10G | FP16 | `/v1/embeddings` |
| `qwen3-vl-embedding-8b` | `Qwen/Qwen3-VL-Embedding-8B` | VL Embed | A10G | FP16 | `/v1/embeddings` |
| `deepseek-ocr-2` | `deepseek-ai/DeepSeek-OCR-2` | OCR | A10G | BF16 | `/v1/chat/completions` |
| `qwen3-tts-0.6b` | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | TTS | A10G | BF16 | `/v1/audio/speech` |
| `qwen3-tts-1.7b` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | TTS | A10G | BF16 | `/v1/audio/speech` |
| `qwen3-asr-0.6b` | `Qwen/Qwen3-ASR-0.6B` | ASR | A10G | BF16 | `/v1/audio/transcriptions` |
| `qwen3-asr-1.7b` | `Qwen/Qwen3-ASR-1.7B` | ASR | A10G | BF16 | `/v1/audio/transcriptions` |

## Installation

**Prerequisites:** Python 3.13+, [uv](https://docs.astral.sh/uv/), [Modal](https://modal.com) account (`modal token new`)

```bash
git clone https://github.com/n24q02m/modalcom-ai-workers.git
cd modalcom-ai-workers
uv sync --all-groups

# Or with mise
mise install && mise run setup
```

## Usage

### Deploy Workers

```bash
# List available models
python -m ai_workers deploy list

# Deploy a single worker
python -m ai_workers deploy qwen3-reranker-8b

# Deploy all workers
python -m ai_workers deploy --all

# Dry run (show what would be deployed)
python -m ai_workers deploy qwen3-reranker-8b --dry-run
```

### ONNX/GGUF Conversion

Convert models to optimized formats and push to HuggingFace Hub. Conversion runs on Modal CPU containers.

```bash
# ONNX (FP32 -> INT8 + Q4F16 quantization)
python -m ai_workers onnx-convert qwen3-embedding-0.6b

# GGUF (Q4_K_M via llama.cpp)
python -m ai_workers gguf-convert qwen3-embedding-0.6b

# List convertible models
python -m ai_workers onnx-convert list
```

### Consumer Usage (via LiteLLM Proxy)

Workers expose OpenAI/Cohere-compatible endpoints. Use [LiteLLM Proxy](https://docs.litellm.ai/) for unified routing, auth, and cost tracking. A ready-to-use config is at [`litellm/config.yaml`](litellm/config.yaml).

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

# Rerank (Cohere-compatible via curl)
# curl -X POST http://localhost:4000/v1/rerank \
#   -H "Authorization: Bearer your-litellm-key" \
#   -d '{"model": "qwen3-reranker-8b", "query": "What is AI?", "documents": ["AI is..."]}'
```

## Configuration

### LiteLLM Model Naming

| Task | LiteLLM Prefix | Example |
|:-----|:---------------|:--------|
| Embedding | `openai/` | `openai/qwen3-embedding-0.6b` |
| Reranker | `cohere/` | `cohere/qwen3-reranker-8b` |
| VL Embedding | `openai/` | `openai/qwen3-vl-embedding-2b` |
| VL Reranker | `cohere/` | `cohere/qwen3-vl-reranker-8b` |
| OCR | `openai/` | `openai/deepseek-ocr-2` |
| TTS | `openai/` | `openai/qwen3-tts-0.6b` |
| ASR | `openai/` | `openai/qwen3-asr-0.6b` |

### Modal Secrets

| Secret Name | Key | Description |
|:------------|:----|:------------|
| `worker-api-key` | `KLPRISM_WORKER_API_KEY` | KnowledgePrism endpoint auth |
| `worker-api-key` | `AIORA_WORKER_API_KEY` | Aiora endpoint auth |
| `hf-token` | `HF_TOKEN` | HuggingFace Hub push (ONNX/GGUF converters) |

### Modal Endpoint URL Format

```
https://<workspace>--<app-name>-<classname-lowercase>-serve.modal.run
```

After deployment, get the exact URL from `modal app list` or the Modal dashboard.

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

## Related Projects

- [qwen3-embed](https://github.com/n24q02m/qwen3-embed) -- Local ONNX runtime for Qwen3 embedding models (offline/edge deployment)
- [wet-mcp](https://github.com/n24q02m/wet-mcp) -- MCP web search server, uses Modal workers for cloud embedding/reranking
- [mnemo-mcp](https://github.com/n24q02m/mnemo-mcp) -- MCP memory server, uses Modal workers for cloud embedding/reranking
- [KnowledgePrism](https://github.com/n24q02m/knowledgeprism) -- Language learning platform, uses Modal workers for RAG pipeline
- [Aiora](https://github.com/n24q02m/aiora) -- Air quality alert platform, uses Modal workers for RAG pipeline

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT -- See [LICENSE](LICENSE).
