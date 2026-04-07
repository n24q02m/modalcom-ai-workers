"""Tests for ai_workers.common.volumes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_download_models_success():
    """Test successful download of all models."""
    mock_targets = ["model-a", "model-b"]

    with (
        patch("ai_workers.common.volumes.ACTIVE_MODEL_HF_IDS", mock_targets),
        patch.dict("sys.modules", {"huggingface_hub": MagicMock()}),
        patch("ai_workers.common.volumes.hf_cache_vol.commit") as mock_commit,
    ):
        import huggingface_hub

        from ai_workers.common.volumes import download_models

        huggingface_hub.snapshot_download.return_value = "/mock/path"

        result = download_models()

        assert huggingface_hub.snapshot_download.call_count == 2
        mock_commit.assert_called_once()
        assert "OK: model-a" in result
        assert "OK: model-b" in result


def test_download_models_partial_failure():
    """Test download where one model fails and one succeeds."""
    mock_targets = ["model-a", "model-fail"]

    def mock_download_side_effect(hf_id, **kwargs):
        if hf_id == "model-fail":
            raise RuntimeError("Download timeout")
        return "/mock/path"

    with (
        patch("ai_workers.common.volumes.ACTIVE_MODEL_HF_IDS", mock_targets),
        patch.dict("sys.modules", {"huggingface_hub": MagicMock()}),
        patch("ai_workers.common.volumes.hf_cache_vol.commit") as mock_commit,
        patch("loguru.logger.error") as mock_logger_error,
    ):
        import huggingface_hub

        from ai_workers.common.volumes import download_models

        huggingface_hub.snapshot_download.side_effect = mock_download_side_effect

        result = download_models()

        assert huggingface_hub.snapshot_download.call_count == 2
        mock_commit.assert_called_once()
        assert "OK: model-a" in result
        assert "FAIL: model-fail (Download timeout)" in result

        # Verify logger.error call
        assert mock_logger_error.call_count == 1
        args, _ = mock_logger_error.call_args
        assert args[0] == "Failed to download {}: {}"
        assert args[1] == "model-fail"
        assert str(args[2]) == "Download timeout"


def test_download_models_all_failure():
    """Test download where all models fail."""
    mock_targets = ["model-a", "model-b"]

    with (
        patch("ai_workers.common.volumes.ACTIVE_MODEL_HF_IDS", mock_targets),
        patch.dict("sys.modules", {"huggingface_hub": MagicMock()}),
        patch("ai_workers.common.volumes.hf_cache_vol.commit") as mock_commit,
        patch("loguru.logger.error") as mock_logger_error,
    ):
        import huggingface_hub

        from ai_workers.common.volumes import download_models

        huggingface_hub.snapshot_download.side_effect = RuntimeError("Network Error")

        result = download_models()

        assert huggingface_hub.snapshot_download.call_count == 2
        mock_commit.assert_called_once()
        assert "FAIL: model-a (Network Error)" in result
        assert "FAIL: model-b (Network Error)" in result

        assert mock_logger_error.call_count == 2
        # Check first failure log
        args0, _ = mock_logger_error.call_args_list[0]
        assert args0[1] == "model-a"
        assert str(args0[2]) == "Network Error"
        # Check second failure log
        args1, _ = mock_logger_error.call_args_list[1]
        assert args1[1] == "model-b"
        assert str(args1[2]) == "Network Error"
