"""Self-hosted AI model workers on Modal.com with LiteLLM-compatible endpoints."""

import contextlib

import modal

with contextlib.suppress(Exception):
    # Initialize Modal (check if local to verify import/init)
    modal.is_local()

__version__ = "0.0.0-dev"
with contextlib.suppress(Exception):
    from importlib.metadata import version

    __version__ = version("ai-workers")
