import sys
from unittest.mock import MagicMock, patch

# Mock modal before importing anything that might use it
sys.modules['modal'] = MagicMock()
# Mock torch and other heavy dependencies
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['loguru'] = MagicMock()

# We need to make sure the VLRerankerServer class itself is not a mock.
# Modal's app.cls() decorator returns a class-like object.
# Let's mock modal.App() to return an object where cls() returns the original class.

mock_app = MagicMock()
def mock_cls_decorator(*args, **kwargs):
    def decorator(cls):
        return cls
    return decorator
mock_app.cls = mock_cls_decorator
sys.modules['modal'].App.return_value = mock_app

import os
sys.path.append(os.getcwd() + "/src")

from ai_workers.workers.vl_reranker import VLRerankerServer

def test_repro():
    server = VLRerankerServer()
    print(f"Type of server: {type(server)}")

    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = "<prompt>"

    mock_inputs = MagicMock()
    mock_inputs.to.return_value = mock_inputs
    mock_processor.return_value = mock_inputs

    mock_hidden = MagicMock()
    # hidden = outputs.last_hidden_state[:, -1, :]
    # Slice [:, -1, :] results in two getitem calls:
    # [:, -1] and then [:] or similar.
    # Actually it's one getitem call with a tuple of slices/ints.
    mock_hidden.__getitem__.return_value = mock_hidden

    mock_outputs = MagicMock()
    mock_outputs.last_hidden_state = mock_hidden

    mock_backbone = MagicMock()
    mock_backbone.return_value = mock_outputs

    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.model = mock_backbone

    mock_yes_no_weight = MagicMock()

    server.models = {"qwen3-vl-reranker-8b": mock_model}
    server.processors = {"qwen3-vl-reranker-8b": mock_processor}
    server.yes_no_weights = {"qwen3-vl-reranker-8b": mock_yes_no_weight}

    with patch('torch.nn.functional.linear') as mock_linear, \
         patch('torch.nn.functional.softmax') as mock_softmax:

        mock_logits = MagicMock()
        mock_linear.return_value = mock_logits

        mock_probs = MagicMock()
        # probs[:, 1].tolist()
        mock_probs.__getitem__.return_value.tolist.return_value = [0.8]
        mock_softmax.return_value = mock_probs

        print("Calling _score_pair...")
        try:
            score = server._score_pair("qwen3-vl-reranker-8b", "query", "document")
            print(f"Score: {score}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_repro()
