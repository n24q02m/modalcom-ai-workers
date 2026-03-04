# Style Guide - modalcom-ai-workers

## Architecture
GPU-serverless AI model workers on Modal.com. Python 3.13, src layout, monorepo with multiple worker modules.

## Python
- Formatter/Linter: Ruff (`line-length = 100`, `target-version = "py313"`)
- Type checker: ty (`src = ["src"]`)
- Test: pytest + pytest-asyncio (`asyncio_mode = "auto"`)
- Package manager: uv
- Runtime: Modal.com (GPU serverless)
- API: FastAPI (per-worker ASGI apps)

## Code Patterns
- Lazy imports for heavy deps (`torch`, `transformers`) — loaded inside functions for Modal container compatibility
- `from __future__ import annotations` in every file
- `@dataclass(frozen=True)` for immutable config objects
- `enum.StrEnum` for all enumerations
- `loguru.logger` for all logging (not stdlib `logging`)
- Auth middleware: Bearer token verification on every worker endpoint
- Worker pattern: `modal.App` → `@app.cls(gpu=...)` → `@modal.enter()` for model loading → `@modal.asgi_app()` for FastAPI

## Worker Architecture
Each worker file follows:
1. Module docstring with LiteLLM integration notes
2. Module-level constants (`SCALEDOWN_WINDOW`, `KEEP_WARM`, `MODEL_CONFIGS`)
3. `modal.App(...)` definition with secrets
4. `@app.cls(gpu=..., image=...)` class with `@modal.concurrent(...)`
5. `@modal.enter()` for model loading at container startup
6. Private compute methods (`_embed`, `_score_pair`)
7. `@modal.asgi_app()` returning FastAPI app with auth middleware + endpoints

## Commits
Conventional Commits (feat:, fix:, chore:, docs:, refactor:, test:).

## Security
Bearer token auth on all worker endpoints. Secrets managed via Modal Secrets (optionally injected via Infisical). No hardcoded credentials.
