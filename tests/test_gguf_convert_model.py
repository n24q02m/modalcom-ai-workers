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
from ai_workers.workers.gguf_converter import gguf_convert_model


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


def test_gguf_convert_model_repo_not_found(mock_hf_hub, mock_env):
    """Test when repo does not exist (list_repo_tree raises Exception)."""
    mock_hf_hub["list_repo_tree"].side_effect = Exception("Repo not found")

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
        )

        assert result["status"] == "success"
        # It should have proceeded to create_repo because list_repo_tree failed
        mock_hf_hub["api"].create_repo.assert_called_once()
        assert mock_run.call_count == 2


def test_gguf_convert_model_config_upload_fail(mock_hf_hub, mock_env):
    """Test when config file upload fails (swallowed Exception)."""
    # Mock upload_file to fail only for the second call (config.json)
    # The first call is for the GGUF file itself.
    # Actually, looking at the code, upload_file is called for GGUF first, then for config files.

    mock_result = MagicMock()
    mock_result.returncode = 0

    with (
        patch("subprocess.run", return_value=mock_result),
        patch("pathlib.Path.stat") as mock_stat,
        patch("pathlib.Path.unlink"),
        patch("pathlib.Path.mkdir"),
        patch("pathlib.Path.resolve", side_effect=lambda: MagicMock()),
        patch("tempfile.TemporaryDirectory") as mock_tmp,
    ):
        mock_tmp.return_value.__enter__.return_value = "/tmp/fake"
        mock_stat.return_value.st_size = 100 * 1024 * 1024

        # Make the second+ upload_file calls fail
        def side_effect(*args, **kwargs):
            if kwargs.get("path_in_repo") in [
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ]:
                raise Exception("Upload failed")
            return MagicMock()

        mock_hf_hub["api"].upload_file.side_effect = side_effect

        result = gguf_convert_model(
            model_name="test-model",
            hf_source="org/source",
            hf_target="org/target-GGUF",
            gguf_name="test",
        )

        assert result["status"] == "success"
        # upload_file should be called for GGUF and attempted for configs
        assert mock_hf_hub["api"].upload_file.call_count >= 1
