"""Tests for onnx_converter module: registry, _generate_model_card, OnnxModelConfig."""

from __future__ import annotations

from ai_workers.workers.onnx_converter import (
    ONNX_MODELS,
    OnnxModelConfig,
    _generate_model_card,
    _register,
)

# ---------------------------------------------------------------------------
# Registry contents
# ---------------------------------------------------------------------------


def test_onnx_models_registry_has_expected_keys():
    assert "qwen3-embedding-0.6b-onnx" in ONNX_MODELS
    assert "qwen3-reranker-0.6b-onnx" in ONNX_MODELS


def test_embedding_config_fields():
    cfg = ONNX_MODELS["qwen3-embedding-0.6b-onnx"]
    assert cfg.model_class == "AutoModel"
    assert cfg.output_attr == "last_hidden_state"
    assert "Qwen3-Embedding" in cfg.hf_source
    assert "ONNX" in cfg.hf_target


def test_reranker_config_fields():
    cfg = ONNX_MODELS["qwen3-reranker-0.6b-onnx"]
    assert cfg.model_class == "AutoModelForCausalLM"
    assert cfg.output_attr == "logits"
    assert "Qwen3-Reranker" in cfg.hf_source
    assert "ONNX" in cfg.hf_target


# ---------------------------------------------------------------------------
# OnnxModelConfig dataclass
# ---------------------------------------------------------------------------


def test_onnx_model_config_frozen():
    cfg = OnnxModelConfig(
        name="test",
        hf_source="org/source",
        hf_target="org/target",
        model_class="AutoModel",
        output_attr="last_hidden_state",
    )
    import pytest

    with pytest.raises((AttributeError, TypeError)):
        cfg.name = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _register
# ---------------------------------------------------------------------------


def test_register_adds_to_registry():
    # Use a unique name not in the base registry
    cfg = OnnxModelConfig(
        name="_test_register_temp",
        hf_source="org/s",
        hf_target="org/t",
        model_class="AutoModel",
        output_attr="last_hidden_state",
    )
    result = _register(cfg)
    assert "_test_register_temp" in ONNX_MODELS
    assert result is cfg
    # Cleanup
    del ONNX_MODELS["_test_register_temp"]


# ---------------------------------------------------------------------------
# _generate_model_card — embedding
# ---------------------------------------------------------------------------


def test_generate_model_card_embedding():
    cfg = OnnxModelConfig(
        name="qwen3-embedding-0.6b-onnx",
        hf_source="Qwen/Qwen3-Embedding-0.6B",
        hf_target="n24q02m/Qwen3-Embedding-0.6B-ONNX",
        model_class="AutoModel",
        output_attr="last_hidden_state",
    )
    card = _generate_model_card(cfg, int8_size_mb=120.5, q4f16_size_mb=80.3)

    assert "text-embedding" in card
    assert "feature-extraction" in card
    assert "TextEmbedding" in card
    assert "120" in card  # int8_size_mb formatted
    assert "80" in card  # q4f16_size_mb formatted
    assert "Qwen/Qwen3-Embedding-0.6B" in card
    assert "n24q02m/Qwen3-Embedding-0.6B-ONNX" in card
    assert "GGUF" in card  # link to GGUF repo


# ---------------------------------------------------------------------------
# _generate_model_card — reranker
# ---------------------------------------------------------------------------


def test_generate_model_card_reranker():
    cfg = OnnxModelConfig(
        name="qwen3-reranker-0.6b-onnx",
        hf_source="Qwen/Qwen3-Reranker-0.6B",
        hf_target="n24q02m/Qwen3-Reranker-0.6B-ONNX",
        model_class="AutoModelForCausalLM",
        output_attr="logits",
    )
    card = _generate_model_card(cfg, int8_size_mb=200.0, q4f16_size_mb=150.0)

    assert "text-reranking" in card
    assert "text-classification" in card
    assert "TextCrossEncoder" in card
    assert "Qwen/Qwen3-Reranker-0.6B" in card
    assert "GGUF" in card


# ---------------------------------------------------------------------------
# _generate_model_card — GGUF link derivation
# ---------------------------------------------------------------------------


def test_generate_model_card_gguf_link():
    cfg = OnnxModelConfig(
        name="test",
        hf_source="org/MyModel",
        hf_target="org/MyModel-ONNX",
        model_class="AutoModel",
        output_attr="last_hidden_state",
    )
    card = _generate_model_card(cfg, int8_size_mb=50.0, q4f16_size_mb=30.0)
    assert "org/MyModel-GGUF" in card


# ---------------------------------------------------------------------------
# onnx_convert_model
# ---------------------------------------------------------------------------


def test_onnx_convert_invalid_model_class(monkeypatch):
    from unittest.mock import MagicMock

    import pytest

    from ai_workers.workers.onnx_converter import onnx_convert_model

    monkeypatch.setenv("HF_TOKEN", "dummy_token")

    # Mock HfApi and repo_exists to prevent actual network calls
    import sys

    mock_hub = MagicMock()
    mock_hub.repo_exists.return_value = False
    sys.modules["huggingface_hub"] = mock_hub

    mock_transformers = MagicMock()
    mock_transformers.AutoTokenizer = MagicMock()
    sys.modules["transformers"] = mock_transformers
    sys.modules["onnx"] = MagicMock()
    sys.modules["onnxruntime"] = MagicMock()
    sys.modules["onnxruntime.quantization"] = MagicMock()
    sys.modules["onnxruntime.quantization.matmul_nbits_quantizer"] = MagicMock()

    # Mock AutoTokenizer to prevent downloading models

    with pytest.raises(ValueError, match=r"Model class 'InvalidClass' is invalid. Choose from:"):
        onnx_convert_model(
            model_name="test_model",
            hf_source="Qwen/Qwen3-Embedding-0.6B",
            hf_target="test/target",
            model_class="InvalidClass",
            output_attr="last_hidden_state",
        )
