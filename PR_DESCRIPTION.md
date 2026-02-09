# 🔧 Fix CI and local setup dependency installation

## 🎯 What
Updated `.github/workflows/ci.yml` and `.mise.toml` to use `uv sync --extra dev` instead of `uv sync --all-groups` or `uv sync --group dev`.

## 🔍 Why
The project uses `[project.optional-dependencies]` (standard Python extras) to define the `dev` environment, not the newer PEP 735 `[dependency-groups]`. `uv sync --all-groups` or `--group dev` looks for dependency groups and does not install extras, causing CI to fail when tools like `ruff` (defined in `dev` extra) are missing.

## ✨ Result
- CI workflow `lint-and-test` now correctly installs `ruff`, `ty`, `pytest`, and other dev dependencies.
- Local setup via `mise install` or `mise run setup` now correctly installs the dev environment.
- Verified locally that `uv sync --extra dev` works and all tests pass (including those requiring `torch` which is now installed via `dev` extra).
