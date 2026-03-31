# modalcom-ai-workers

GPU-serverless AI model workers deployed on Modal.com.

## Tech Stack

- **Runtime**: Python 3.13, Modal (GPU serverless)
- **Package Manager**: uv
- **Framework**: FastAPI (LiteLLM-compatible endpoints)
- **Testing**: pytest, pytest-asyncio, pytest-cov
- **Linting**: ruff, ty
- **Release**: python-semantic-release (Conventional Commits)

## Project Structure

- `src/ai_workers/workers/` -- individual worker modules (embedding, reranking, OCR, ASR, TTS, VL)
- `src/ai_workers/common/` -- shared utilities, volumes, config
- `src/ai_workers/cli/` -- deployment and model conversion CLI
- `tests/` -- unit tests per worker and module

## Key Conventions

- Worker naming: `test_workers_{type}.py` mirrors `workers/{type}.py`
- All workers expose LiteLLM-compatible HTTP endpoints
- Auth: per-app Bearer tokens (WORKER_API_KEY_{APP})
- GPU containers use A10G, scale-to-zero after 5 minutes
- ONNX conversion tooling for INT8 quantization
- Infisical for secrets management (project: 53c9e228)
- Coverage target: >= 95%

## Commands

```bash
# Run tests
uv run pytest tests/ -v --cov
# Deploy a worker
uv run python -m ai_workers.cli deploy --worker <name>
# ONNX conversion
uv run python -m ai_workers.cli onnx-convert --model <hf_id>
```
