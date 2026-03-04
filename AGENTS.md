# AGENTS.md - modalcom-ai-workers

GPU-serverless AI model workers on Modal.com. Python 3.13, uv, src layout.

## Build / Lint / Test Commands

```bash
uv sync --all-groups                # Install all dependency groups
uv run ruff check .                 # Lint
uv run ruff format --check .        # Format check
uv run ruff check --fix .           # Lint fix
uv run ruff format .                # Format fix
uv run ty check                     # Type check (Astral ty, src/ only)
uv run pytest                       # Run all tests
uv run pytest tests/ -v --tb=short  # CI test command

# Run a single test file
uv run pytest tests/test_config.py

# Run a single test function
uv run pytest tests/test_config.py::test_function_name -v

# Mise shortcuts
mise run setup       # Full dev setup
mise run lint        # ruff check + ruff format --check + ty check
mise run test        # pytest
mise run fix         # ruff check --fix + ruff format

# Deployment (requires Infisical secrets)
mise run deploy      # Deploy single worker (interactive)
mise run deploy-all  # Deploy all workers
```

### Pytest Configuration

- `asyncio_mode = "auto"` -- no `@pytest.mark.asyncio` needed
- `testpaths = ["tests"]`

## Code Style

### Formatting (Ruff)

- **Line length**: 100
- **Quotes**: Double quotes
- **Indent**: 4 spaces
- **Target**: Python 3.13

### Ruff Rules

`select = ["E", "W", "F", "I", "N", "UP", "B", "SIM", "TCH", "RUF"]`
`ignore = ["E501", "B008"]`

- `N` = pep8-naming, `TCH` = type-checking imports, `RUF` = ruff-specific rules
- `B008` ignored (function calls in defaults -- needed for Typer)
- `known-first-party = ["ai_workers"]`

### Type Checker (ty)

`src = ["src"]`, `python-version = "3.13"`

### Import Ordering

1. `from __future__ import annotations` (used in nearly every file)
2. Standard library
3. Third-party (`modal`, `fastapi`, `loguru`, `pydantic`)
4. Local (`ai_workers.common.config`, `ai_workers.common.auth`)

**Lazy imports**: Heavy deps (`torch`, `transformers`, `PIL`) imported INSIDE functions, not at module level. This is deliberate for Modal container compatibility and fast CLI startup.

**`TYPE_CHECKING` guard** for type-only imports.

```python
from __future__ import annotations

import hmac
import os

from fastapi import HTTPException, Request, status
from loguru import logger

from ai_workers.common.config import ModelConfig
```

### Type Hints

- Full type hints on all signatures
- Modern syntax: `str | None`, `list[str]`, `dict[str, object]`
- `@dataclass(frozen=True)` for immutable config objects
- `enum.StrEnum` for all enumerations

### Naming Conventions

| Element            | Convention       | Example                            |
|--------------------|------------------|------------------------------------|
| Functions/methods  | snake_case       | `verify_api_key`, `load_models`    |
| Private methods    | `_snake_case`    | `_embed`, `_score_pair`            |
| Classes            | PascalCase       | `ModelConfig`, `EmbeddingServer`   |
| Constants          | UPPER_SNAKE_CASE | `SCALEDOWN_WINDOW`, `MODEL_CONFIGS` |
| Enums              | PascalCase/UPPER | `Task.EMBEDDING`, `GPU.A10G`       |
| Modules            | snake_case       | `vl_embedding.py`, `vl_reranker.py` |

### Error Handling

- `msg = f"..."; raise ExceptionType(msg)` -- assign message first (Ruff B-rules)
- `HTTPException` with `status_code` + `detail` for API errors
- `loguru.logger` for all logging (not stdlib `logging`)
- Auth middleware: try/except HTTPException, return JSONResponse on failure

### Worker Architecture Pattern

Every worker file follows this structure:
1. Module docstring with LiteLLM integration notes
2. `from __future__ import annotations` + `import modal`
3. Module-level constants (`SCALEDOWN_WINDOW`, `KEEP_WARM`, `MODEL_CONFIGS`)
4. `modal.App(...)` definition with secrets
5. `@app.cls(gpu=..., image=...)` + `@modal.concurrent(...)` class
6. `@modal.enter()` method for model loading at container startup
7. Private compute methods (`_embed`, `_score_pair`)
8. `@modal.asgi_app()` method returning FastAPI app with:
   - Inline Pydantic request/response models
   - Auth middleware (copy-pasted pattern)
   - `/health` GET + main POST endpoint

### File Organization

```
src/ai_workers/
  __init__.py, __main__.py    # Package + CLI entry
  common/                     # Shared infrastructure
    config.py                 # Model registry (single source of truth)
    auth.py                   # Bearer token middleware
    images.py                 # Modal container image builders
    logging.py                # Structured logging config
  cli/                        # Typer CLI commands (deploy, onnx-convert, gguf-convert)
  workers/                    # Modal.com worker definitions
    embedding.py, reranker.py, vl_embedding.py, vl_reranker.py, ocr.py, asr.py
    onnx_converter.py, gguf_converter.py
tests/                        # Unit tests
```

### Documentation

- Module-level docstrings on every file
- Google-style docstrings: `Args:`, `Returns:` sections
- All comments and docstrings in English

### Commits

Conventional Commits: `type(scope): message`.

### Pre-commit Hooks

1. Ruff lint (`--fix --target-version=py313`) + format
2. ty type check
3. pytest (`--tb=short -q`)
