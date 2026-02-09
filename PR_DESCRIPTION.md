# 🔧 Fix `ty` configuration in `pyproject.toml`

## 🎯 What
Updated `[tool.ty]` configuration in `pyproject.toml` to use `environment = { python = ".venv/bin/python3" }` instead of the invalid `python-version = "3.13"`.

## 🔍 Why
The previous configuration caused `ty` to fail with a TOML parse error: `unknown field 'python-version'`. The correct field for specifying the python environment in `ty` is `environment`. Additionally, pointing it to the virtualenv python executable ensures `ty` can resolve installed dependencies correctly.

## ✨ Result
- `ty check` now runs successfully without configuration errors.
- CI workflow `lint-and-test` should now proceed past the type check step.
