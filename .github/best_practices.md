# Style Guide - modalcom-ai-workers

## Architecture
GPU-serverless AI model workers on Modal.com. Python, src-layout single-package repo.

## Python
- Formatter/Linter: Ruff (target Python 3.13)
- Type checker: ty
- Test: pytest (232 tests, mocks for modal/torch/transformers)
- Package manager: uv
- Core deps: fastapi, pydantic, loguru
- Runtime deps (Modal containers only): modal, torch, transformers, huggingface-hub

## Code Patterns
- Model registry in `common/config.py` — single source of truth for all 10 models
- Bearer token auth middleware (`common/auth.py`) — timing-safe comparison
- Modal container images built in `common/images.py` — shared across workers
- Light + Heavy model variants merged into single Modal apps
- Lazy imports for ML deps (modal, torch, transformers) — not available locally
- Custom mean-pool + L2-normalize for embeddings (not using model's built-in)
- Yes/no logit scoring for rerankers via CausalLM

## Commits
Conventional Commits: only `feat:` and `fix:` allowed (enforced by pre-commit hook).

## Security
All secrets via environment variables or Modal Secrets. No hardcoded keys. Worker endpoints protected by bearer token auth.
