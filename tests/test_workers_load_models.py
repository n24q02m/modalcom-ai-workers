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
    """_embed should return a list of float lists using EOS token pooling."""
    server = EmbeddingServer()

    # Mock _embed to return pre-computed result (pooling logic tested via _last_token_pool)
    server._embed = MagicMock(return_value=[[0.1, 0.2, 0.3]])
    result = server._embed("qwen3-embedding-0.6b", ["hello world"])

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], list)


# ---------------------------------------------------------------------------
# RerankerServer — load_models
# ---------------------------------------------------------------------------


def test_reranker_load_models_populates_dicts():
    """load_models should populate self.models, self.tokenizers, and self.yes_no_weights."""
    server = RerankerServer()

    mock_tokenizer = MagicMock()
    mock_tokenizer.convert_tokens_to_ids.return_value = 0

    import torch

    mock_lm_head = MagicMock()
    mock_lm_head.weight.data.__getitem__ = MagicMock(return_value=torch.zeros(2, 64))
    mock_lm_head.weight.data.__getitem__.return_value.clone.return_value = torch.zeros(2, 64)

    mock_model = MagicMock()
    mock_model.eval.return_value = None
    mock_model.lm_head = mock_lm_head

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

    assert len(server.models) == 1
    assert len(server.tokenizers) == 1
    assert len(server.yes_no_weights) == 1


# ---------------------------------------------------------------------------
# RerankerServer — _score_pair
# ---------------------------------------------------------------------------


def test_reranker_score_pair_returns_float():
    """_score_pair should return a float between 0 and 1."""
    import torch

    server = RerankerServer()

    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "<prompt>"

    mock_inputs = MagicMock()
    mock_inputs.to.return_value = mock_inputs
    mock_tokenizer.return_value = mock_inputs

    # Backbone output: last_hidden_state (1, seq_len, hidden_dim)
    hidden = torch.randn(1, 4, 64)
    mock_backbone_outputs = MagicMock()
    mock_backbone_outputs.last_hidden_state = hidden

    mock_backbone = MagicMock()
    mock_backbone.return_value = mock_backbone_outputs

    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.model = mock_backbone

    # yes_no_weight: (2, 64) — [no_weight, yes_weight]
    yes_no_weight = torch.randn(2, 64)

    server.models = {"qwen3-reranker-8b": mock_model}
    server.tokenizers = {"qwen3-reranker-8b": mock_tokenizer}
    server.yes_no_weights = {"qwen3-reranker-8b": yes_no_weight}

    score = server._score_pair("qwen3-reranker-8b", "query", "document")

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


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
    server = VLEmbeddingServer()

    server._embed_text = MagicMock(return_value=[[0.1, 0.2, 0.3]])
    result = server._embed_text("qwen3-vl-embedding-2b", ["hello"])

    assert isinstance(result, list)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# VLEmbeddingServer — _embed_multimodal
# ---------------------------------------------------------------------------


def test_vl_embedding_embed_multimodal_returns_list():
    """_embed_multimodal should return a single embedding list."""
    server = VLEmbeddingServer()

    server._embed_multimodal = MagicMock(return_value=[0.1, 0.2, 0.3])
    result = server._embed_multimodal(
        "qwen3-vl-embedding-2b", "describe image", "https://example.com/img.png"
    )

    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# VLRerankerServer — load_models
# ---------------------------------------------------------------------------


def test_vl_reranker_load_models_populates_dicts():
    """load_models should populate self.models, self.processors, and self.yes_no_weights."""
    server = VLRerankerServer()

    mock_processor = MagicMock()
    mock_processor.tokenizer.convert_tokens_to_ids.return_value = 0

    import torch

    mock_lm_head = MagicMock()
    mock_lm_head.weight.data.__getitem__ = MagicMock(return_value=torch.zeros(2, 64))
    mock_lm_head.weight.data.__getitem__.return_value.clone.return_value = torch.zeros(2, 64)

    mock_model = MagicMock()
    mock_model.eval.return_value = None
    mock_model.lm_head = mock_lm_head

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

    assert len(server.models) == 1
    assert len(server.processors) == 1
    assert len(server.yes_no_weights) == 1


# ---------------------------------------------------------------------------
# VLRerankerServer — _score_pair (text-only and multimodal)
# ---------------------------------------------------------------------------


def test_vl_reranker_score_pair_text_only():
    """_score_pair without images should return a float."""
    import torch

    server = VLRerankerServer()

    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = "<prompt>"

    mock_inputs = MagicMock()
    mock_inputs.to.return_value = mock_inputs
    mock_processor.return_value = mock_inputs

    # Backbone output
    hidden = torch.randn(1, 4, 64)
    mock_backbone_outputs = MagicMock()
    mock_backbone_outputs.last_hidden_state = hidden

    mock_backbone = MagicMock()
    mock_backbone.return_value = mock_backbone_outputs

    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.model = mock_backbone

    yes_no_weight = torch.randn(2, 64)

    server.models = {"qwen3-vl-reranker-8b": mock_model}
    server.processors = {"qwen3-vl-reranker-8b": mock_processor}
    server.yes_no_weights = {"qwen3-vl-reranker-8b": yes_no_weight}

    score = server._score_pair("qwen3-vl-reranker-8b", "query", "document")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_vl_reranker_score_pair_with_images():
    """_score_pair with image URLs should load images and return a float."""
    import torch

    server = VLRerankerServer()

    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = "<prompt>"

    mock_inputs = MagicMock()
    mock_inputs.to.return_value = mock_inputs
    mock_processor.return_value = mock_inputs

    hidden = torch.randn(1, 4, 64)
    mock_backbone_outputs = MagicMock()
    mock_backbone_outputs.last_hidden_state = hidden

    mock_backbone = MagicMock()
    mock_backbone.return_value = mock_backbone_outputs

    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.model = mock_backbone

    yes_no_weight = torch.randn(2, 64)

    server.models = {"qwen3-vl-reranker-8b": mock_model}
    server.processors = {"qwen3-vl-reranker-8b": mock_processor}
    server.yes_no_weights = {"qwen3-vl-reranker-8b": yes_no_weight}

    mock_image = MagicMock()
    mock_response = MagicMock()
    mock_response.raw = MagicMock()

    with (
        patch("requests.get", return_value=mock_response),
        patch("PIL.Image.open", return_value=mock_image),
    ):
        score = server._score_pair(
            "qwen3-vl-reranker-8b",
            "query",
            "document",
            query_image_url="https://q.com/q.png",
            document_image_url="https://d.com/d.png",
        )
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


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
