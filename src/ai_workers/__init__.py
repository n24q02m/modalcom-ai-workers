"""Self-hosted AI model workers on Modal.com with LiteLLM-compatible endpoints."""

try:
    from importlib.metadata import version

    __version__ = version("ai-workers")
except Exception:
    # Package metadata not available in Modal containers (add_local_python_source)
    __version__ = "0.0.0-dev"
