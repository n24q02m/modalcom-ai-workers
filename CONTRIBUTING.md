# Contributing to modalcom-ai-workers

Thank you for your interest in contributing! This guide will help you get started.

## Prerequisites

- [Python](https://www.python.org/) 3.13+
- [uv](https://docs.astral.sh/uv/)
- [mise](https://mise.jdx.dev/) (recommended)
- [Modal](https://modal.com/) account (for deployment testing)

## Setup

```bash
git clone https://github.com/n24q02m/modalcom-ai-workers.git
cd modalcom-ai-workers
mise run setup    # or: uv sync --group dev
```

## Development Workflow

1. Create a branch from `main`:
   ```bash
   git checkout -b feat/my-feature
   ```

2. Make your changes and test:
   ```bash
   uv run pytest --tb=short          # Unit tests
   uv run ruff check .               # Lint
   uv run ruff format --check .      # Format check
   uv run ty check                   # Type check
   ```

3. Commit using only `feat:` or `fix:` prefixes (enforced):
   ```
   feat: add new ASR language support
   fix: correct batch size calculation for reranker
   ```

4. Push and open a Pull Request against `main`

## Project Structure

```
src/ai_workers/
  __init__.py               # Package init (dynamic version)
  __main__.py               # CLI entrypoint
  images.py                 # Modal image definitions
  workers/                  # One file per worker type
    embedding.py            # Text embedding (ONNX, A10G)
    reranker.py             # Text reranking (ONNX, A10G)
    vl_embedding.py         # Vision-language embedding
    vl_reranker.py          # Vision-language reranking
    ocr.py                  # OCR (Docling)
    asr.py                  # ASR (Whisper)
litellm/
  config.yaml               # LiteLLM proxy config
tests/
  test_*.py                 # Unit tests
```

## Code Style

- **Formatter**: [Ruff](https://docs.astral.sh/ruff/) (4-space indent, double quotes, 100 line width)
- **Linting**: Ruff rules (E, W, F, I, N, UP, B, SIM, RUF)
- **Type checker**: [ty](https://docs.astral.sh/ty/)
- **Target**: Python 3.13

## Adding a New Worker

1. Create `src/ai_workers/workers/<name>.py` with a Modal `App` and FastAPI endpoint
2. Define the Modal image in `images.py` if new dependencies are needed
3. Add the worker URL to `litellm/config.yaml`
4. Write tests in `tests/test_<name>.py`
5. Update the worker matrix table in `README.md`

## Testing

- Write unit tests for all new functionality
- Place tests in `tests/` directory
- Mock Modal and model calls — no GPU required for tests

```bash
uv run pytest                    # All tests
uv run pytest --tb=short -q      # Short output
```

## Pull Request Guidelines

- Fill out the PR template completely
- Ensure all CI checks pass
- Keep PRs focused on a single concern
- Update `README.md` if adding/changing workers or configuration
- Add tests for new functionality

## Release Process

Releases are automated via [python-semantic-release](https://python-semantic-release.readthedocs.io/)
and triggered manually through the CD workflow. Version bumps are determined by commit messages
(`feat:` → minor, `fix:` → patch).

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
