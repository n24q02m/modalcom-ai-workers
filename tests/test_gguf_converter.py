"""Tests for gguf_converter module: registry, _generate_gguf_model_card, GgufModelConfig."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ai_workers.workers.gguf_converter import (
    GGUF_MODELS,
    GgufModelConfig,
    _generate_gguf_model_card,
    _register,
    gguf_convert_model,
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


# ---------------------------------------------------------------------------
# gguf_convert_model — repo checking error path
# ---------------------------------------------------------------------------


def test_gguf_convert_model_repo_not_found_swallows_exception():
    """Verify that gguf_convert_model continues if list_repo_tree fails (repo missing)."""
    # Mock huggingface_hub
    mock_hf_hub = MagicMock()
    mock_hf_hub.list_repo_tree.side_effect = Exception("Repo not found")

    # Mock subprocess.run to avoid executing llama.cpp
    mock_run = MagicMock()
    mock_run.returncode = 0

    # Mock HfApi
    mock_api_instance = MagicMock()
    mock_hf_hub.HfApi.return_value = mock_api_instance

    with (
        patch.dict("sys.modules", {"huggingface_hub": mock_hf_hub}),
        patch("subprocess.run", return_value=mock_run),
        patch("os.environ", {"HF_TOKEN": "fake-token"}),
        patch("pathlib.Path.stat") as mock_stat,
        patch("pathlib.Path.unlink"),
        patch("ai_workers.workers.gguf_converter._generate_gguf_model_card", return_value="card"),
    ):
        # stat().st_size for f16_path and q4_path
        mock_stat.return_value.st_size = 100 * 1024 * 1024

        result = gguf_convert_model(
            model_name="qwen3-embedding-0.6b-gguf",
            hf_source="Qwen/Qwen3-Embedding-0.6B",
            hf_target="n24q02m/Qwen3-Embedding-0.6B-GGUF",
            gguf_name="qwen3-embedding-0.6b",
        )

    assert result["status"] == "success"
    # Verify list_repo_tree was called and exception was swallowed
    mock_hf_hub.list_repo_tree.assert_called_once()
    # Verify it continued to create_repo and upload_file
    mock_api_instance.create_repo.assert_called_once()
    mock_api_instance.upload_file.assert_called()


# ---------------------------------------------------------------------------
# _check_if_gguf_exists — exception handling
# ---------------------------------------------------------------------------


def test_check_if_gguf_exists_exception_handling():
    """Verify that _check_if_gguf_exists returns False and swallows Exception."""
    mock_hf = MagicMock()
    mock_hf.list_repo_tree.side_effect = Exception("HF Hub Error")

    with patch.dict("sys.modules", {"huggingface_hub": mock_hf}):
        from ai_workers.workers.gguf_converter import _check_if_gguf_exists

        result = _check_if_gguf_exists(
            hf_target="org/repo",
            gguf_repo_path="model.gguf",
            hf_token="fake-token",
        )

        assert result is False
        mock_hf.list_repo_tree.assert_called_with("org/repo", token="fake-token", recursive=True)
