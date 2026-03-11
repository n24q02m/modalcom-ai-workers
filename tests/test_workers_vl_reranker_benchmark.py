import os
import time
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from ai_workers.workers.vl_reranker import VLRerankerServer


def test_benchmark():
    server = VLRerankerServer()
    server.models = {"qwen3-vl-reranker-2b": MagicMock()}
    server.processors = {"qwen3-vl-reranker-2b": MagicMock()}

    def slow_load_image(url):
        time.sleep(0.1)  # Simulate 100ms network latency
        return MagicMock()

    server._load_image = MagicMock(side_effect=slow_load_image)

    # Instead of full torch execution, let's mock torch out
    import builtins

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "torch":
            mock_torch = MagicMock()
            mock_torch.no_grad = MagicMock()
            # mock sigmoid to return an item
            mock_item = MagicMock()
            mock_item.item.return_value = 0.5
            mock_torch.sigmoid.return_value = mock_item
            return mock_torch
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        app = server.serve()

    tc = TestClient(app)

    start = time.time()
    with patch.dict(os.environ, {"API_KEY": "test"}):
        resp = tc.post(
            "/v1/rerank",
            json={
                "model": "qwen3-vl-reranker-2b",
                "query": "find dogs",
                "query_image_url": "http://example.com/query.jpg",
                "documents": [
                    "dog 1",
                    "dog 2",
                    "cat 1",
                    "cat 2",
                    "bird 1",
                ],
            },
            headers={"Authorization": "Bearer test"},
        )
    end = time.time()

    assert resp.status_code == 200, resp.json()
    print(f"\n[BENCHMARK] Baseline execution time for 5 documents: {end - start:.4f}s")
    print(f"[BENCHMARK] Number of times `_load_image` was called: {server._load_image.call_count}")
