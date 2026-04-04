from unittest.mock import MagicMock, patch
import pytest
import sys
from pathlib import Path

# Mock huggingface_hub to avoid ModuleNotFoundError in environment
sys.modules["huggingface_hub"] = MagicMock()

from ai_workers.workers.gguf_converter import gguf_convert_model

@patch("ai_workers.workers.gguf_converter._check_if_gguf_exists")
@patch("ai_workers.workers.gguf_converter._download_hf_model")
@patch("ai_workers.workers.gguf_converter._convert_hf_to_f16_gguf")
@patch("ai_workers.workers.gguf_converter._quantize_f16_to_gguf")
@patch("ai_workers.workers.gguf_converter._upload_gguf_artifacts")
@patch("tempfile.TemporaryDirectory")
@patch("os.environ.get")
@patch("pathlib.Path.unlink")
@patch("pathlib.Path.mkdir")
def test_gguf_convert_model_orchestration(
    mock_mkdir,
    mock_unlink,
    mock_env_get,
    mock_tmpdir,
    mock_upload,
    mock_quantize,
    mock_convert,
    mock_download,
    mock_exists,
):
    # Setup mocks
    mock_env_get.return_value = "fake-token"
    mock_exists.return_value = False
    mock_tmpdir.return_value.__enter__.return_value = "/tmp/fake"
    mock_convert.return_value = 1000.0  # f16_size
    mock_quantize.return_value = 250.0  # q4_size

    # Execute
    result = gguf_convert_model(
        model_name="test-model",
        hf_source="org/source",
        hf_target="org/target",
        gguf_name="test",
        force=False
    )

    # Verify orchestration
    mock_exists.assert_called_once()
    mock_download.assert_called_once()
    mock_convert.assert_called_once()
    mock_quantize.assert_called_once()
    mock_upload.assert_called_once()
    mock_unlink.assert_called_once()

    assert result["status"] == "success"
    assert result["size_mb"] == 250.0

@patch("ai_workers.workers.gguf_converter._check_if_gguf_exists")
@patch("os.environ.get")
def test_gguf_convert_model_skips_if_exists(mock_env_get, mock_exists):
    mock_env_get.return_value = "fake-token"
    mock_exists.return_value = True

    result = gguf_convert_model(
        model_name="test-model",
        hf_source="org/source",
        hf_target="org/target",
        gguf_name="test",
        force=False
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "already_exists"
