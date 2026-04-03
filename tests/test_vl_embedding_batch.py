# ruff: noqa: E402
import os
import sys
from unittest.mock import MagicMock

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

# Mock torch
mock_torch = MagicMock()
sys.modules["torch"] = mock_torch
sys.modules["torch.nn"] = MagicMock()
sys.modules["torch.nn.functional"] = mock_torch.nn.functional

# Mock modal before importing VLEmbeddingServer
mock_modal = MagicMock()
sys.modules["modal"] = mock_modal
sys.modules["ai_workers.common.images"] = MagicMock()
sys.modules["ai_workers.common.volumes"] = MagicMock()
sys.modules["ai_workers.common.config"] = MagicMock()


# Mock the class decorator to return the class itself
def mock_decorator(*args, **kwargs):
    def wrapper(cls):
        return cls

    return wrapper


mock_modal.App.return_value.cls = mock_decorator
mock_modal.concurrent = mock_decorator

# Import vl_embedding first to set constants
import ai_workers.workers.vl_embedding

ai_workers.workers.vl_embedding.EMBEDDING_DIM = 1024
ai_workers.workers.vl_embedding.DEFAULT_INSTRUCTION = "Represent the user's input."

from ai_workers.workers.vl_embedding import VLEmbeddingServer


def test_embed_text_is_batched():
    # Setup server with mocks
    server = VLEmbeddingServer()
    mock_model = MagicMock()
    mock_processor = MagicMock()

    model_name = "qwen3-vl-embedding-2b"
    server.models = {model_name: mock_model}
    server.processors = {model_name: mock_processor}

    # Mock model.device
    mock_model.device = "cpu"

    # Mock apply_chat_template to return a list of strings
    texts = ["text1", "text2", "text3"]
    mock_processor.apply_chat_template.return_value = ["formatted1", "formatted2", "formatted3"]

    # Mock processor call to return tensors
    batch_size = len(texts)

    mock_inputs = {
        "input_ids": MagicMock(),
        "attention_mask": MagicMock(),
    }

    class MockInputs(dict):
        def to(self, device):
            return self

    mock_processor.return_value = MockInputs(mock_inputs)

    # Mock model forward pass
    mock_outputs = MagicMock()
    mock_outputs.last_hidden_state = MagicMock()
    mock_model.return_value = mock_outputs

    # Mock _last_token_pool to return a mocked tensor
    mock_batched_embeddings = MagicMock()
    # Mock the slicing
    mock_batched_embeddings.__getitem__.return_value = mock_batched_embeddings
    mock_batched_embeddings.cpu.return_value.tolist.return_value = [[0.1] * 1024] * batch_size

    server._last_token_pool = MagicMock(return_value=mock_batched_embeddings)
    mock_torch.nn.functional.normalize.return_value = mock_batched_embeddings

    # Call _embed_text
    embeddings = server._embed_text(model_name, texts)

    # Assertions
    assert isinstance(embeddings, list)
    assert len(embeddings) == batch_size
    assert len(embeddings[0]) == 1024

    # Verify batched calls
    mock_processor.apply_chat_template.assert_called_once()
    mock_processor.assert_called_once()
    mock_model.assert_called_once()

    # Verify arguments to processor
    _, kwargs = mock_processor.call_args
    assert kwargs["text"] == ["formatted1", "formatted2", "formatted3"]


if __name__ == "__main__":
    test_embed_text_is_batched()
