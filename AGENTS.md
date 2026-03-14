# AGENTS.md - modalcom-ai-workers

GPU-serverless AI workers on Modal.com for reranking, embedding, OCR, TTS, and ASR. Python >= 3.13, uv, Modal.

## Build / Lint / Test Commands

```bash
uv sync --group dev                # Install dependencies
uv run ruff check .                # Lint
uv run ruff format --check .       # Format check
uv run ruff format .               # Format fix
uv run ruff check --fix .          # Lint fix
uv run ty check                    # Type check (Astral ty)

# Tests
uv run pytest                      # All tests
uv run pytest --tb=short -q        # Short output

# Run a single test file
uv run pytest tests/test_embedding.py

# Run a single test function
uv run pytest tests/test_embedding.py::test_function_name -v

# Modal deployment (requires MODAL_TOKEN_ID + MODAL_TOKEN_SECRET)
modal deploy src/ai_workers/workers/embedding.py

# Mise shortcuts
mise run setup     # Full dev environment setup
mise run lint      # ruff check + ruff format --check + ty check
mise run test      # pytest
mise run fix       # ruff check --fix + ruff format
```

### Pytest Configuration

- `testpaths = ["tests"]`, `pythonpath = ["src"]`
- No integration markers — all tests are unit tests (workers mocked)

## Code Style

### Formatting (Ruff)

- **Line length**: 100
- **Quotes**: Double quotes
- **Indent**: 4 spaces
- **Target**: Python 3.13

### Ruff Rules

`select = ["E", "W", "F", "I", "N", "UP", "B", "SIM", "RUF"]`, `ignore = ["E501", "B008"]`

- `I` = isort, `N` = pep8-naming, `UP` = pyupgrade, `B` = bugbear, `SIM` = simplify, `RUF` = ruff-specific

### Type Checker (ty)

Uses defaults (no custom config beyond `python-version = "3.13"`).

### Import Ordering (isort via Ruff)

1. Standard library (`import json`, `from pathlib import Path`)
2. Third-party (`import modal`, `from fastapi import FastAPI`)
3. Local (`from ai_workers.workers.embedding import ...`)

```python
import json
from pathlib import Path

import modal
from fastapi import FastAPI

from ai_workers.workers.embedding import EmbeddingWorker
```

### Type Hints

- Full type hints everywhere: parameters, return types, variables
- Union types: `str | None` (not `Optional`), `list[str]` (not `List`)
- `from __future__ import annotations` where needed for forward refs

### Naming Conventions

| Element            | Convention       | Example                          |
|--------------------|------------------|----------------------------------|
| Functions/methods  | snake_case       | `embed_text`, `rerank_documents` |
| Private methods    | `_snake_case`    | `_load_model`, `_preprocess`     |
| Classes            | PascalCase       | `EmbeddingWorker`, `OcrWorker`   |
| Constants          | UPPER_SNAKE_CASE | `MODEL_NAME`, `MAX_BATCH_SIZE`   |
| Modules/packages   | snake_case       | `embedding.py`, `vl_reranker.py` |

### Error Handling

- `ValueError` for input/config validation
- `raise ... from e` for exception chaining
- `loguru.logger` for logging

### File Organization

```
src/ai_workers/
  __init__.py                    # Dynamic version (importlib.metadata)
  __main__.py                    # CLI entrypoint
  images.py                      # Modal image definitions (shared deps)
  workers/
    __init__.py
    embedding.py                 # Text embedding worker (ONNX)
    reranker.py                  # Text reranker worker (Qwen3-Reranker-8B)
    vl_embedding.py              # Vision-language embedding worker
    vl_reranker.py               # Vision-language reranker worker (Qwen3-VL-Reranker-8B)
    ocr.py                       # OCR worker (DeepSeek-OCR-2)
    tts.py                       # TTS worker (Qwen3-TTS)
    asr.py                       # ASR worker (Qwen3-ASR)
    onnx_converter.py            # ONNX model conversion worker
    gguf_converter.py            # GGUF model conversion worker
litellm/
  config.yaml                    # LiteLLM proxy config referencing these workers
  README.md                      # LiteLLM integration guide
tests/
  test_*.py                      # Unit tests (workers mocked)
```

### Documentation

- Google-style docstrings with `Args:`, `Returns:`, `Raises:` sections
- Public API and complex methods must have docstrings

### Commits

Only `feat:` and `fix:` prefixes allowed (enforced via pre-commit `commit-msg` hook).

### Pre-commit Hooks

1. Ruff lint (`--fix --target-version=py313`) + format
2. ty type check
3. pytest (`--tb=short -q`)
