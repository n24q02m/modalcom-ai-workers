import sys
import pytest
from unittest.mock import MagicMock, patch

# Define mocks
mock_modal = MagicMock()
mock_app = MagicMock()
def cls_decorator(*args, **kwargs):
    def wrapper(cls):
        return cls
    return wrapper
mock_app.cls.side_effect = cls_decorator
mock_modal.App.return_value = mock_app
mock_modal.asgi_app.return_value = lambda f: f
mock_modal.enter.return_value = lambda f: f

# Mock FastAPI
mock_fastapi = MagicMock()
captured_handlers = {}
def post_decorator(path, **kwargs):
    def decorator(func):
        captured_handlers[path] = func
        return func
    return decorator
mock_fastapi.FastAPI.return_value.post.side_effect = post_decorator
mock_fastapi.FastAPI.return_value.get.side_effect = lambda path, **kwargs: lambda func: func
mock_fastapi.FastAPI.return_value.middleware.return_value = lambda f: f

# Mock Pydantic
mock_pydantic = MagicMock()
class MockBaseModel:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
mock_pydantic.BaseModel = MockBaseModel

@pytest.mark.asyncio
async def test_rerank_batching():
    # Setup sys.modules patches
    patches = {
        "modal": mock_modal,
        "ai_workers.common.images": MagicMock(),
        "ai_workers.common.r2": MagicMock(),
        "ai_workers.common.auth": MagicMock(),
        "fastapi": mock_fastapi,
        "pydantic": mock_pydantic,
    }

    with patch.dict(sys.modules, patches):
        # Force reload or fresh import
        if "ai_workers.workers.vl_reranker" in sys.modules:
            del sys.modules["ai_workers.workers.vl_reranker"]

        from ai_workers.workers import vl_reranker

        # Instantiate
        server = vl_reranker.VLRerankerLightServer()

        # Spy on _score_batch
        # We use a list to store call count to mutate it from inner function
        call_stats = {"count": 0}

        def spied_score_batch(query, documents):
            call_stats["count"] += 1
            return [0.95] * len(documents)

        server._score_batch = spied_score_batch

        # Mock other dependencies
        server.processor = MagicMock()
        server.model = MagicMock()

        # Get app and handler
        # Reset captured handlers
        captured_handlers.clear()

        app = server.serve()
        rerank_handler = captured_handlers.get("/v1/rerank")

        assert rerank_handler is not None

        # Create request
        class MockRequest(MockBaseModel):
            pass

        request = MockRequest(
            query="test query",
            documents=[f"doc {i}" for i in range(10)],
            top_n=None,
            model="test",
            return_documents=True
        )

        # Run handler
        await rerank_handler(request)

        # Verify
        assert call_stats["count"] == 1, f"Expected 1 call to _score_batch, got {call_stats['count']}"

if __name__ == "__main__":
    # Allow running as script too (install pytest-asyncio or just run with pytest)
    sys.exit(pytest.main(["-v", __file__]))
