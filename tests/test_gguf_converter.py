"""Tests for gguf_converter module: registry, _generate_gguf_model_card, GgufModelConfig."""

from __future__ import annotations

import pytest

from ai_workers.workers.gguf_converter import (
    GGUF_MODELS,
    GgufModelConfig,
    _generate_gguf_model_card,
    _register,
)


# ---------------------------------------------------------------------------
# Registry contents
# ---------------------------------------------------------------------------


def test_gguf_models_registry_has_expected_keys():
    assert "qwen3-embedding-0.6b-gguf" in GGUF_MODELS
    assert "qwen3-reranker-0.6b-gguf" in GGUF_MODELS


def test_embedding_config_fields():
    cfg = GGUF_MODELS["qwen3-embedding-0.6b-gguf"]
    assert cfg.output_attr == "last_hidden_state"
    assert "Qwen3-Embedding" in cfg.hf_source
    assert "GGUF" in cfg.hf_target
    assert cfg.gguf_name == "qwen3-embedding-0.6b"


def test_reranker_config_fields():
    cfg = GGUF_MODELS["qwen3-reranker-0.6b-gguf"]
    assert cfg.output_attr == "logits"
    assert "Qwen3-Reranker" in cfg.hf_source
    assert "GGUF" in cfg.hf_target
    assert cfg.gguf_name == "qwen3-reranker-0.6b"


# ---------------------------------------------------------------------------
# GgufModelConfig dataclass
# ---------------------------------------------------------------------------


def test_gguf_model_config_frozen():
    cfg = GgufModelConfig(
        name="test",
        hf_source="org/source",
        hf_target="org/target-GGUF",
        gguf_name="test-model",
        output_attr="last_hidden_state",
    )
    with pytest.raises((AttributeError, TypeError)):
        cfg.name = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _register
# ---------------------------------------------------------------------------


def test_register_adds_to_registry():
    cfg = GgufModelConfig(
        name="_test_gguf_register_temp",
        hf_source="org/s",
        hf_target="org/t-GGUF",
        gguf_name="test",
        output_attr="last_hidden_state",
    )
    result = _register(cfg)
    assert "_test_gguf_register_temp" in GGUF_MODELS
    assert result is cfg
    # Cleanup
    del GGUF_MODELS["_test_gguf_register_temp"]


# ---------------------------------------------------------------------------
# _generate_gguf_model_card — embedding
# ---------------------------------------------------------------------------


def test_generate_gguf_model_card_embedding():
    cfg = GgufModelConfig(
        name="qwen3-embedding-0.6b-gguf",
        hf_source="Qwen/Qwen3-Embedding-0.6B",
        hf_target="n24q02m/Qwen3-Embedding-0.6B-GGUF",
        gguf_name="qwen3-embedding-0.6b",
        output_attr="last_hidden_state",
    )
    card = _generate_gguf_model_card(
        cfg,
        gguf_filename="qwen3-embedding-0.6b-q4-k-m.gguf",
        size_mb=250.0,
    )

    assert "text-embedding" in card
    assert "feature-extraction" in card
    assert "TextEmbedding" in card
    assert "qwen3-embedding-0.6b-q4-k-m.gguf" in card
    assert "250" in card
    assert "Qwen/Qwen3-Embedding-0.6B" in card
    assert "ONNX" in card  # link to ONNX repo


# ---------------------------------------------------------------------------
# _generate_gguf_model_card — reranker
# ---------------------------------------------------------------------------


def test_generate_gguf_model_card_reranker():
    cfg = GgufModelConfig(
        name="qwen3-reranker-0.6b-gguf",
        hf_source="Qwen/Qwen3-Reranker-0.6B",
        hf_target="n24q02m/Qwen3-Reranker-0.6B-GGUF",
        gguf_name="qwen3-reranker-0.6b",
        output_attr="logits",
    )
    card = _generate_gguf_model_card(
        cfg,
        gguf_filename="qwen3-reranker-0.6b-q4-k-m.gguf",
        size_mb=300.0,
    )

    assert "text-reranking" in card
    assert "text-classification" in card
    assert "TextCrossEncoder" in card
    assert "ONNX" in card


# ---------------------------------------------------------------------------
# _generate_gguf_model_card — ONNX link derivation
# ---------------------------------------------------------------------------


def test_generate_gguf_model_card_onnx_link():
    cfg = GgufModelConfig(
        name="test",
        hf_source="org/MyModel",
        hf_target="org/MyModel-GGUF",
        gguf_name="my-model",
        output_attr="last_hidden_state",
    )
    card = _generate_gguf_model_card(cfg, gguf_filename="my-model-q4-k-m.gguf", size_mb=100.0)
    assert "org/MyModel-ONNX" in card
