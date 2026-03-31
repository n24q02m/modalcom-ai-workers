"""Tests for ai_workers.common.volumes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ai_workers.common.volumes import download_models


def test_download_models_success():
    """Test successful download of all models."""
    mock_targets = ["model-a", "model-b"]

    with (
        patch("ai_workers.common.volumes.ACTIVE_MODEL_HF_IDS", mock_targets),
        patch.dict("sys.modules", {"huggingface_hub": MagicMock()}),
        patch("ai_workers.common.volumes.hf_cache_vol.commit") as mock_commit,
    ):
        # We also need to patch loguru locally since we aren't pulling it from sys.modules
        from ai_workers.common.volumes import download_models as dl_func
        # Let's import huggingface_hub here so it pulls the mocked version
        import huggingface_hub
        huggingface_hub.snapshot_download.return_value = "/mock/path"

        result = dl_func()

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
    ):
        from ai_workers.common.volumes import download_models as dl_func
        import huggingface_hub
        huggingface_hub.snapshot_download.side_effect = mock_download_side_effect

        result = dl_func()

        assert huggingface_hub.snapshot_download.call_count == 2
        mock_commit.assert_called_once()
        assert "OK: model-a" in result
        assert "FAIL: model-fail (Download timeout)" in result


def test_download_models_all_failure():
    """Test download where all models fail."""
    mock_targets = ["model-a", "model-b"]

    with (
        patch("ai_workers.common.volumes.ACTIVE_MODEL_HF_IDS", mock_targets),
        patch.dict("sys.modules", {"huggingface_hub": MagicMock()}),
        patch("ai_workers.common.volumes.hf_cache_vol.commit") as mock_commit,
    ):
        from ai_workers.common.volumes import download_models as dl_func
        import huggingface_hub
        huggingface_hub.snapshot_download.side_effect = RuntimeError("Network Error")

        result = dl_func()

        assert huggingface_hub.snapshot_download.call_count == 2
        mock_commit.assert_called_once()
        assert "FAIL: model-a (Network Error)" in result
        assert "FAIL: model-b (Network Error)" in result
