"""Unit tests for gguf_convert_model in src/ai_workers/workers/gguf_converter.py."""

from __future__ import annotations

import os
import re
import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock huggingface_hub in sys.modules before any imports or patches
mock_hf_mod = MagicMock()
sys.modules["huggingface_hub"] = mock_hf_mod

# ruff: noqa: E402
from ai_workers.workers.gguf_converter import (
    GGUF_MODELS,
    GgufModelConfig,
    _generate_gguf_model_card,
    _register,
    gguf_convert_model,
)


@pytest.fixture
def mock_hf_hub():
    # We can patch the attributes of the already-mocked module
    with (
        patch("huggingface_hub.HfApi") as mock_api_cls,
        patch("huggingface_hub.list_repo_tree") as mock_list_repo,
        patch("huggingface_hub.snapshot_download") as mock_snapshot,
        patch("huggingface_hub.hf_hub_download") as mock_hf_download,
    ):
        mock_api = mock_api_cls.return_value
        yield {
            "api": mock_api,
            "list_repo_tree": mock_list_repo,
            "snapshot_download": mock_snapshot,
            "hf_hub_download": mock_hf_download,
        }


@pytest.fixture
def mock_env():
    with patch.dict(os.environ, {"HF_TOKEN": "fake-token"}):
        yield


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
# _generate_gguf_model_card
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
# gguf_convert_model
# ---------------------------------------------------------------------------


def test_gguf_convert_model_success(mock_hf_hub, mock_env):
    """Test successful conversion and upload."""
    mock_result = MagicMock()
    mock_result.returncode = 0

    with (
        patch("subprocess.run", return_value=mock_result) as mock_run,
        patch("pathlib.Path.stat") as mock_stat,
        patch("pathlib.Path.unlink"),
        patch("pathlib.Path.mkdir"),
        patch("pathlib.Path.resolve", side_effect=lambda: MagicMock()),
        patch("tempfile.TemporaryDirectory") as mock_tmp,
    ):
        mock_tmp.return_value.__enter__.return_value = "/tmp/fake"
        mock_stat.return_value.st_size = 100 * 1024 * 1024  # 100 MB

        result = gguf_convert_model(
            model_name="test-model",
            hf_source="org/source",
            hf_target="org/target-GGUF",
            gguf_name="test",
            output_attr="last_hidden_state",
        )

        assert result["status"] == "success"
        assert result["hf_target"] == "org/target-GGUF"
        assert result["size_mb"] == 100.0

        # Verify HF API calls
        mock_hf_hub["api"].create_repo.assert_called_once()
        assert mock_hf_hub["api"].upload_file.call_count >= 2

        # Verify subprocess calls
        assert mock_run.call_count == 2


def test_gguf_convert_model_skipped(mock_hf_hub, mock_env):
    """Test skipping when file already exists."""
    mock_file = MagicMock()
    mock_file.path = "test-q4-k-m.gguf"
    mock_hf_hub["list_repo_tree"].return_value = [mock_file]

    result = gguf_convert_model(
        model_name="test-model",
        hf_source="org/source",
        hf_target="org/target-GGUF",
        gguf_name="test",
        quant_type="Q4_K_M",
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "already_exists"

    # Ensure no download/convert happened
    mock_hf_hub["snapshot_download"].assert_not_called()


def test_gguf_convert_model_invalid_inputs(mock_env):
    """Test validation of gguf_name and quant_type."""
    with pytest.raises(ValueError, match="Invalid gguf_name"):
        gguf_convert_model("m", "s", "t", "invalid name!")

    with pytest.raises(ValueError, match="Invalid quant_type"):
        gguf_convert_model("m", "s", "t", "valid", quant_type="invalid type!")


def test_gguf_convert_model_missing_token():
    """Test error when HF_TOKEN is missing."""
    with (
        patch.dict(os.environ, {}, clear=True),
        pytest.raises(ValueError, match="HF_TOKEN is not set"),
    ):
        gguf_convert_model("m", "s", "t", "v")


def test_gguf_convert_model_convert_fail(mock_hf_hub, mock_env):
    """Test failure during Step 1 (F16 conversion)."""
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "error in conversion"

    with (
        patch("subprocess.run", return_value=mock_result),
        patch("pathlib.Path.mkdir"),
        patch("tempfile.TemporaryDirectory"),
        pytest.raises(RuntimeError, match=re.escape("convert_hf_to_gguf.py failed")),
    ):
        gguf_convert_model("m", "s", "t", "v")


def test_gguf_convert_model_quantize_fail(mock_hf_hub, mock_env):
    """Test failure during Step 2 (quantization)."""
    mock_res_success = MagicMock(returncode=0)
    mock_res_fail = MagicMock(returncode=1, stderr="error in quantize")

    with (
        patch("subprocess.run", side_effect=[mock_res_success, mock_res_fail]),
        patch("pathlib.Path.stat") as mock_stat,
        patch("pathlib.Path.mkdir"),
        patch("tempfile.TemporaryDirectory"),
    ):
        mock_stat.return_value.st_size = 100
        with pytest.raises(RuntimeError, match=re.escape("llama-quantize failed")):
            gguf_convert_model("m", "s", "t", "v")


def test_gguf_convert_model_repo_not_found_swallows_exception(mock_hf_hub, mock_env):
    """Verify that gguf_convert_model continues if list_repo_tree fails (repo missing)."""
    mock_hf_hub["list_repo_tree"].side_effect = Exception("Repo not found")

    mock_run = MagicMock()
    mock_run.returncode = 0

    with (
        patch("subprocess.run", return_value=mock_run),
        patch("pathlib.Path.stat") as mock_stat,
        patch("pathlib.Path.unlink"),
        patch("pathlib.Path.mkdir"),
        patch("pathlib.Path.resolve", side_effect=lambda: MagicMock()),
        patch("tempfile.TemporaryDirectory") as mock_tmp,
        patch("ai_workers.workers.gguf_converter._generate_gguf_model_card", return_value="card"),
    ):
        mock_tmp.return_value.__enter__.return_value = "/tmp/fake"
        mock_stat.return_value.st_size = 100 * 1024 * 1024

        result = gguf_convert_model(
            model_name="qwen3-embedding-0.6b-gguf",
            hf_source="Qwen/Qwen3-Embedding-0.6B",
            hf_target="n24q02m/Qwen3-Embedding-0.6B-GGUF",
            gguf_name="qwen3-embedding-0.6b",
        )

    assert result["status"] == "success"
    mock_hf_hub["list_repo_tree"].assert_called_once()
    mock_hf_hub["api"].create_repo.assert_called_once()


def test_gguf_convert_model_upload_config_fail_swallows_exception(mock_hf_hub, mock_env):
    """Verify that failures in uploading optional config files are swallowed (lines 331-332)."""
    # hf_hub_download fails for one of the config files
    mock_hf_hub["hf_hub_download"].side_effect = [
        "/tmp/config.json",
        Exception("Failed to download tokenizer.json"),
        "/tmp/tokenizer_config.json",
    ]

    mock_run = MagicMock()
    mock_run.returncode = 0

    with (
        patch("subprocess.run", return_value=mock_run),
        patch("pathlib.Path.stat") as mock_stat,
        patch("pathlib.Path.unlink"),
        patch("pathlib.Path.mkdir"),
        patch("pathlib.Path.resolve", side_effect=lambda: MagicMock()),
        patch("tempfile.TemporaryDirectory") as mock_tmp,
        patch("ai_workers.workers.gguf_converter._generate_gguf_model_card", return_value="card"),
    ):
        mock_tmp.return_value.__enter__.return_value = "/tmp/fake"
        mock_stat.return_value.st_size = 100 * 1024 * 1024

        result = gguf_convert_model(
            model_name="test-model",
            hf_source="org/source",
            hf_target="org/target-GGUF",
            gguf_name="test",
        )

    assert result["status"] == "success"
    # Should have called hf_hub_download 3 times (it tries all files in the loop)
    assert mock_hf_hub["hf_hub_download"].call_count == 3
