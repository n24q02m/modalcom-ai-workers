import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Mock google.generativeai and tenacity if they are missing
try:
    import google.generativeai
except ImportError:
    mock_genai = MagicMock()
    sys.modules["google"] = MagicMock()
    sys.modules["google.generativeai"] = mock_genai
    sys.modules["tenacity"] = MagicMock()

# Add training to sys.path to import hard_negative_mining
training_path = str(Path(__file__).parent.parent / "training" / "gemma4_reranker")
if training_path not in sys.path:
    sys.path.insert(0, training_path)

import unittest

import numpy as np
from data_pipeline import TrainSample
from hard_negative_mining import GeminiMiner, MiningTask


class TestGeminiMiner(unittest.TestCase):
    def setUp(self):
        # We need to ensure HAS_GENAI is True for the tests to run the logic
        import hard_negative_mining

        hard_negative_mining.HAS_GENAI = True

        # Mock the generative model
        with patch("google.generativeai.GenerativeModel"):
            self.miner = GeminiMiner(api_key="fake_key")
            self.miner.teacher_client = MagicMock()

    def test_process_query_sample(self):
        # Mock embed_content and teacher score
        with (
            patch.object(self.miner, "embed_content", return_value=np.array([0.1, 0.2, 0.3])),
            patch.object(self.miner, "get_teacher_score", return_value=0.9),
        ):
            task = MiningTask(
                query="What is AI?",
                positive="AI is artificial intelligence.",
                corpus=[
                    "AI is cool.",
                    "Dogs are animals.",
                    "Python is a language.",
                    "AI is smart.",
                    "Computers use AI.",
                    "Learning AI.",
                    "AI models.",
                    "Neural networks.",
                    "Machine learning.",
                    "Deep learning.",
                    "AI research.",
                    "AI ethics.",
                    "AI safety.",
                ],
                modality="text",
            )

            result = self.miner.process_query_sample(task)

            self.assertIsInstance(result, TrainSample)
            self.assertEqual(result.query, "What is AI?")
            self.assertEqual(result.positive, "AI is artificial intelligence.")
            self.assertIsInstance(result.negatives, list)
            self.assertEqual(result.modality, "text")
            self.assertEqual(result.teacher_pos_score, 0.9)
            self.assertIsInstance(result.teacher_neg_scores, list)
            self.assertEqual(len(result.teacher_neg_scores), len(result.negatives))

    def test_mining_task_pydantic(self):
        task = MiningTask(query="q", positive="p", corpus=["c1", "c2"])
        self.assertEqual(task.query, "q")
        self.assertEqual(task.modality, "text")  # default


if __name__ == "__main__":
    unittest.main()
