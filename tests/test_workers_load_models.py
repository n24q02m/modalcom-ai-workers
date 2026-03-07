"""Tests for worker load_models() and compute methods (_embed, _score_pair, etc.).

These tests call the methods directly with pre-populated mock objects to cover
the container-startup code that cannot be reached via HTTP routes.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ai_workers.workers.embedding import EmbeddingServer
from ai_workers.workers.reranker import RerankerServer
from ai_workers.workers.vl_embedding import VLEmbeddingServer
from ai_workers.workers.vl_reranker import VLRerankerServer

# ---------------------------------------------------------------------------
# EmbeddingServer — load_models
# ---------------------------------------------------------------------------


def test_embedding_load_models_populates_dicts():
    """load_models should populate self.models and self.tokenizers for all configs."""
    server = EmbeddingServer()

    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_model.eval.return_value = None

    mock_auto_tokenizer = MagicMock(return_value=mock_tokenizer)
    mock_auto_model = MagicMock()
    mock_auto_model.from_pretrained.return_value = mock_model

    import torch

    with (
        patch.dict(
            "sys.modules",
            {
                "transformers": MagicMock(
                    AutoModel=mock_auto_model,
                    AutoTokenizer=mock_auto_tokenizer,
                ),
            },
        ),
        patch("torch.float16", torch.float16),
    ):
        server.load_models()

    assert hasattr(server, "models")
    assert hasattr(server, "tokenizers")
    assert len(server.models) == 2
    assert len(server.tokenizers) == 2


# ---------------------------------------------------------------------------
# EmbeddingServer — _embed
# ---------------------------------------------------------------------------


def test_embedding_embed_returns_correct_shape():
    """_embed should return a list of float lists."""
    import torch

    server = EmbeddingServer()

    # Build mock model with a realistic output
    mock_model = MagicMock()
    mock_model.device = "cpu"

    # Simulate tokenizer output with real torch tensors
    attention_mask = torch.ones(1, 4, dtype=torch.float32)
    mock_inputs = MagicMock()
    mock_inputs.__getitem__ = lambda self, key: (
        attention_mask if key == "attention_mask" else MagicMock()
    )
    mock_inputs.to.return_value = mock_inputs

    last_hidden = torch.randn(1, 4, 64)
    mock_outputs = MagicMock()
    mock_outputs.last_hidden_state = last_hidden

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = mock_inputs

    mock_model.return_value = mock_outputs

    server.models = {"qwen3-embedding-0.6b": mock_model}
    server.tokenizers = {"qwen3-embedding-0.6b": mock_tokenizer}

    result = server._embed("qwen3-embedding-0.6b", ["hello world"])

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], list)


# ---------------------------------------------------------------------------
# RerankerServer — load_models
# ---------------------------------------------------------------------------


def test_reranker_load_models_populates_dicts():
    """load_models should populate self.models and self.tokenizers."""
    server = RerankerServer()

    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_model.eval.return_value = None

    import torch

    with (
        patch.dict(
            "sys.modules",
            {
                "transformers": MagicMock(
                    AutoModelForCausalLM=MagicMock(
                        from_pretrained=MagicMock(return_value=mock_model)
                    ),
                    AutoTokenizer=MagicMock(from_pretrained=MagicMock(return_value=mock_tokenizer)),
                ),
            },
        ),
        patch("torch.float16", torch.float16),
    ):
        server.load_models()

    assert len(server.models) == 2
    assert len(server.tokenizers) == 2


# ---------------------------------------------------------------------------
# RerankerServer — _score_pair
# ---------------------------------------------------------------------------


def test_reranker_score_batch_returns_list_of_floats():
    """_score_batch should return a list of floats between 0 and 1."""
    from unittest.mock import MagicMock

    # We mock torch globally so we can control arange
    mock_torch = MagicMock()

    # Mock inputs logic
    mock_inputs = MagicMock()
    mock_inputs.__getitem__.return_value = MagicMock()
    mock_inputs.__getitem__.return_value.sum.return_value = MagicMock()
    mock_inputs.__getitem__.return_value.sum.return_value.__sub__.return_value = MagicMock()

    # Logits: yes_id=1, no_id=2 → logits[1]=2.0, logits[2]=0.0 → sigmoid(2.0)≈0.88
    mock_outputs = MagicMock()
    mock_outputs.logits = MagicMock()
    # Mocking logits[:, yes_id] -> [2.0]
    mock_outputs.logits.__getitem__.return_value = mock_outputs.logits
    mock_outputs.logits.float.return_value = mock_outputs.logits

    # Mock sigmoid
    mock_torch.sigmoid.return_value = MagicMock()
    mock_torch.sigmoid.return_value.cpu.return_value.tolist.return_value = [0.88]

    with patch.dict("sys.modules", {"torch": mock_torch}):
        from ai_workers.workers.reranker import RerankerServer

        server = RerankerServer()

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "<prompt>"
        mock_tokenizer.convert_tokens_to_ids.side_effect = lambda t: 1 if t == "yes" else 2
        mock_tokenizer_return = MagicMock()
        mock_tokenizer_return.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_tokenizer_return

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.return_value = mock_outputs

        server.models = {"qwen3-reranker-0.6b": mock_model}
        server.tokenizers = {"qwen3-reranker-0.6b": mock_tokenizer}

        scores = server._score_batch("qwen3-reranker-0.6b", "query", ["document"])

        assert isinstance(scores, list)
        assert len(scores) == 1
        assert isinstance(scores[0], float)
        assert scores[0] == 0.88


# ---------------------------------------------------------------------------
# VLEmbeddingServer — load_models
# ---------------------------------------------------------------------------


def test_vl_embedding_load_models_populates_dicts():
    """load_models should populate self.models and self.processors."""
    server = VLEmbeddingServer()

    mock_processor = MagicMock()
    mock_model = MagicMock()
    mock_model.eval.return_value = None

    import torch

    with (
        patch.dict(
            "sys.modules",
            {
                "transformers": MagicMock(
                    AutoModel=MagicMock(from_pretrained=MagicMock(return_value=mock_model)),
                    AutoProcessor=MagicMock(from_pretrained=MagicMock(return_value=mock_processor)),
                ),
            },
        ),
        patch("torch.float16", torch.float16),
    ):
        server.load_models()

    assert len(server.models) == 2
    assert len(server.processors) == 2


# ---------------------------------------------------------------------------
# VLEmbeddingServer — _embed_text
# ---------------------------------------------------------------------------


def test_vl_embedding_embed_text_returns_list():
    """_embed_text should return a list of embeddings."""
    import torch

    server = VLEmbeddingServer()

    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = "<formatted>"
    attention_mask = torch.ones(1, 4, dtype=torch.float32)
    mock_inputs = MagicMock()
    mock_inputs.__getitem__ = lambda self, key: (
        attention_mask if key == "attention_mask" else MagicMock()
    )
    mock_inputs.to.return_value = mock_inputs
    mock_processor.return_value = mock_inputs

    last_hidden = torch.randn(1, 4, 64)
    mock_outputs = MagicMock()
    mock_outputs.last_hidden_state = last_hidden

    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.return_value = mock_outputs

    server.models = {"qwen3-vl-embedding-2b": mock_model}
    server.processors = {"qwen3-vl-embedding-2b": mock_processor}

    result = server._embed_text("qwen3-vl-embedding-2b", ["hello"])

    assert isinstance(result, list)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# VLEmbeddingServer — _embed_multimodal
# ---------------------------------------------------------------------------


def test_vl_embedding_embed_multimodal_returns_list():
    """_embed_multimodal should return a single embedding list."""
    import torch

    server = VLEmbeddingServer()

    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = "<formatted>"
    attention_mask = torch.ones(1, 4, dtype=torch.float32)
    mock_inputs = MagicMock()
    mock_inputs.__getitem__ = lambda self, key: (
        attention_mask if key == "attention_mask" else MagicMock()
    )
    mock_inputs.to.return_value = mock_inputs
    mock_processor.return_value = mock_inputs

    last_hidden = torch.randn(1, 4, 64)
    mock_outputs = MagicMock()
    mock_outputs.last_hidden_state = last_hidden

    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.return_value = mock_outputs

    server.models = {"qwen3-vl-embedding-2b": mock_model}
    server.processors = {"qwen3-vl-embedding-2b": mock_processor}

    mock_image = MagicMock()
    mock_response = MagicMock()
    mock_response.raw = MagicMock()

    with (
        patch("requests.get", return_value=mock_response),
        patch("PIL.Image.open", return_value=mock_image),
    ):
        result = server._embed_multimodal(
            "qwen3-vl-embedding-2b", "describe image", "https://example.com/img.png"
        )

    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# VLRerankerServer — load_models
# ---------------------------------------------------------------------------


def test_vl_reranker_load_models_populates_dicts():
    """load_models should populate self.models and self.processors."""
    server = VLRerankerServer()

    mock_processor = MagicMock()
    mock_model = MagicMock()
    mock_model.eval.return_value = None

    import torch

    with (
        patch.dict(
            "sys.modules",
            {
                "transformers": MagicMock(
                    AutoModelForImageTextToText=MagicMock(
                        from_pretrained=MagicMock(return_value=mock_model)
                    ),
                    AutoProcessor=MagicMock(from_pretrained=MagicMock(return_value=mock_processor)),
                ),
            },
        ),
        patch("torch.float16", torch.float16),
    ):
        server.load_models()

    assert len(server.models) == 2
    assert len(server.processors) == 2


# ---------------------------------------------------------------------------
# VLRerankerServer — _score_pair (text-only and multimodal)
# ---------------------------------------------------------------------------


def test_vl_reranker_score_pair_text_only():
    """_score_pair without images should return a float."""
    import torch

    server = VLRerankerServer()

    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = "<prompt>"
    mock_processor.tokenizer.convert_tokens_to_ids.side_effect = lambda t: 1 if t == "yes" else 2

    mock_inputs = MagicMock()
    mock_inputs.to.return_value = mock_inputs
    mock_processor.return_value = mock_inputs

    logits_tensor = torch.tensor([0.0, 2.0, 0.0])
    mock_outputs = MagicMock()
    mock_outputs.logits.__getitem__.return_value = logits_tensor

    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.return_value = mock_outputs

    server.models = {"qwen3-vl-reranker-2b": mock_model}
    server.processors = {"qwen3-vl-reranker-2b": mock_processor}

    score = server._score_pair("qwen3-vl-reranker-2b", "query", "document")
    assert isinstance(score, float)


def test_vl_reranker_score_pair_with_images():
    """_score_pair with image URLs should load images and return a float."""
    import torch

    server = VLRerankerServer()

    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = "<prompt>"
    mock_processor.tokenizer.convert_tokens_to_ids.side_effect = lambda t: 1 if t == "yes" else 2

    mock_inputs = MagicMock()
    mock_inputs.to.return_value = mock_inputs
    mock_processor.return_value = mock_inputs

    logits_tensor = torch.tensor([0.0, 1.0, 0.0])
    mock_outputs = MagicMock()
    mock_outputs.logits.__getitem__.return_value = logits_tensor

    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.return_value = mock_outputs

    server.models = {"qwen3-vl-reranker-2b": mock_model}
    server.processors = {"qwen3-vl-reranker-2b": mock_processor}

    mock_image = MagicMock()
    mock_response = MagicMock()
    mock_response.raw = MagicMock()

    with (
        patch("requests.get", return_value=mock_response),
        patch("PIL.Image.open", return_value=mock_image),
    ):
        score = server._score_pair(
            "qwen3-vl-reranker-2b",
            "query",
            "document",
            query_image_url="https://q.com/q.png",
            document_image_url="https://d.com/d.png",
        )
    assert isinstance(score, float)


# ---------------------------------------------------------------------------
# VLRerankerServer — _load_image
# ---------------------------------------------------------------------------


def test_vl_reranker_load_image():
    """_load_image should call requests.get and return a PIL Image."""
    mock_image = MagicMock()
    mock_response = MagicMock()
    mock_response.raw = MagicMock()

    with (
        patch("requests.get", return_value=mock_response) as mock_get,
        patch("PIL.Image.open", return_value=mock_image),
    ):
        result = VLRerankerServer._load_image("https://example.com/img.jpg")

    mock_get.assert_called_once_with("https://example.com/img.jpg", stream=True, timeout=30)
    assert result is mock_image
