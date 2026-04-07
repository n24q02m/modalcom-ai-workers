"""Self-hosted AI model workers on Modal.com with LiteLLM-compatible endpoints."""

import contextlib

with contextlib.suppress(Exception):
    import modal

    modal.is_local()

with contextlib.suppress(Exception):
    from importlib.metadata import version

    __version__ = version("ai-workers")

if "__version__" not in globals():
    # Package metadata not available in Modal containers (add_local_python_source)
    __version__ = "0.0.0-dev"
