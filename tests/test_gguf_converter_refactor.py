"""Tests for GGUF converter orchestration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ai_workers.workers.gguf_converter import gguf_convert_model


@patch("ai_workers.workers.gguf_converter.subprocess.run")
@patch("tempfile.TemporaryDirectory")
@patch("os.environ.get")
@patch("pathlib.Path.unlink")
@patch("pathlib.Path.mkdir")
@patch("pathlib.Path.stat")
def test_gguf_convert_model_orchestration(
    mock_stat,
    mock_mkdir,
    mock_unlink,
    mock_env_get,
    mock_tmpdir,
    mock_run,
):
    # Setup mocks
    mock_env_get.return_value = "fake-token"
    mock_tmpdir.return_value.__enter__.return_value = "/tmp/fake"

    # Mock subprocess.run
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_run.return_value = mock_proc

    mock_stat.return_value.st_size = 1024 * 1024 * 250

    # Mock huggingface_hub
    mock_hf = MagicMock()
    mock_hf.list_repo_tree.return_value = []
    mock_hf.snapshot_download.return_value = None
    mock_hf.hf_hub_download.return_value = "fake-local-cfg"

    with patch.dict("sys.modules", {"huggingface_hub": mock_hf, "modal": MagicMock()}):
        # Execute
        result = gguf_convert_model(
            model_name="test-model",
            hf_source="org/source",
            hf_target="org/target",
            gguf_name="test",
            force=False,
        )

    # Verify orchestration
    assert result["status"] == "success"
    assert result["size_mb"] == 250.0
    mock_hf.list_repo_tree.assert_called()
    mock_hf.snapshot_download.assert_called()
    assert mock_run.call_count == 2
    mock_hf.HfApi.return_value.upload_file.assert_called()


@patch("ai_workers.workers.gguf_converter.subprocess.run")
@patch("os.environ.get")
def test_gguf_convert_model_skips_if_exists(mock_env_get, mock_run):
    mock_env_get.return_value = "fake-token"

    # Mock huggingface_hub
    mock_hf = MagicMock()
    mock_file = MagicMock()
    mock_file.path = "test-q4-k-m.gguf"
    mock_hf.list_repo_tree.return_value = [mock_file]

    with patch.dict("sys.modules", {"huggingface_hub": mock_hf, "modal": MagicMock()}):
        result = gguf_convert_model(
            model_name="test-model",
            hf_source="org/source",
            hf_target="org/target",
            gguf_name="test",
            force=False,
        )

    assert result["status"] == "skipped"
    assert result["reason"] == "already_exists"
    mock_run.assert_not_called()
