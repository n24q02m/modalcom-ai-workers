"""Modal.com worker definitions.

Each worker module contains one or more Modal apps with @modal.asgi_app()
endpoints. Workers are deployed individually via `modal deploy`.
"""

__all__ = [
    "asr",
    "embedding",
    "ocr",
    "reranker",
    "vl_embedding",
    "vl_reranker",
]
