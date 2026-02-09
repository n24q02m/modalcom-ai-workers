import sys
import unittest
from unittest.mock import MagicMock

from fastapi import FastAPI

# Mock modal before importing reranker
sys.modules["modal"] = MagicMock()

# Mock ai_workers.common modules
sys.modules["ai_workers.common.images"] = MagicMock()
sys.modules["ai_workers.common.r2"] = MagicMock()
sys.modules["ai_workers.common.auth"] = MagicMock()

# Now import the module
# We need to ensure that when 'ai_workers.workers.reranker' is imported,
# it can access 'ai_workers' package. Assuming PYTHONPATH includes src.

from ai_workers.workers.reranker import (  # noqa: E402
    MODEL_HEAVY,
    MODEL_LIGHT,
    RerankRequestHeavy,
    RerankRequestLight,
    create_reranker_app,
)


class TestRerankerRefactor(unittest.TestCase):
    def test_create_app(self):
        score_fn = MagicMock(return_value=0.9)
        app = create_reranker_app(
            title="Test App",
            model_name="test-model",
            request_model=RerankRequestLight,
            score_fn=score_fn,
        )
        self.assertIsInstance(app, FastAPI)
        self.assertEqual(app.title, "Test App")

    def test_request_models(self):
        req_light = RerankRequestLight(query="q", documents=["d"])
        self.assertEqual(req_light.model, MODEL_LIGHT)

        req_heavy = RerankRequestHeavy(query="q", documents=["d"])
        self.assertEqual(req_heavy.model, MODEL_HEAVY)


if __name__ == "__main__":
    unittest.main()
