"""Tests for ai_workers.common.volumes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


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

    def mock_download_side_effect(repo_id, **kwargs):
        if repo_id == "model-fail":
            raise RuntimeError("Download timeout")
        return "/mock/path"

    with (
        patch("ai_workers.common.volumes.ACTIVE_MODEL_HF_IDS", mock_targets),
        patch.dict("sys.modules", {"huggingface_hub": MagicMock()}),
        patch("ai_workers.common.volumes.hf_cache_vol.commit") as mock_commit,
    ):
        import huggingface_hub

        from ai_workers.common.volumes import download_models

        huggingface_hub.snapshot_download.side_effect = mock_download_side_effect

        with pytest.raises(RuntimeError, match="Download timeout"):
            download_models()

        assert huggingface_hub.snapshot_download.call_count == 2
        # Commit should NOT be called on failure
        mock_commit.assert_not_called()


def test_download_models_all_failure():
    """Test download where all models fail."""
    mock_targets = ["model-a", "model-b"]

    with (
        patch("ai_workers.common.volumes.ACTIVE_MODEL_HF_IDS", mock_targets),
        patch.dict("sys.modules", {"huggingface_hub": MagicMock()}),
        patch("ai_workers.common.volumes.hf_cache_vol.commit") as mock_commit,
    ):
        import huggingface_hub

        from ai_workers.common.volumes import download_models

        huggingface_hub.snapshot_download.side_effect = RuntimeError("Network Error")

        with pytest.raises(RuntimeError, match="Network Error"):
            download_models()

        assert huggingface_hub.snapshot_download.call_count == 1  # Should stop after first failure
        mock_commit.assert_not_called()


def test_download_models_logs_error():
    """Test that download failure is logged and re-raised."""
    mock_targets = ["model-fail"]

    with (
        patch("ai_workers.common.volumes.ACTIVE_MODEL_HF_IDS", mock_targets),
        patch.dict("sys.modules", {"huggingface_hub": MagicMock()}),
        patch("ai_workers.common.volumes.hf_cache_vol.commit"),
        patch("loguru.logger.error") as mock_logger_error,
    ):
        import huggingface_hub

        from ai_workers.common.volumes import download_models

        huggingface_hub.snapshot_download.side_effect = RuntimeError("Something went wrong")

        with pytest.raises(RuntimeError, match="Something went wrong"):
            download_models()

        mock_logger_error.assert_called_once()
        args, _ = mock_logger_error.call_args
        # Loguru positional formatting: (format_string, *args)
        assert args[0] == "Failed to download {}: {}"
        assert args[1] == "model-fail"
        assert "Something went wrong" in str(args[2])
