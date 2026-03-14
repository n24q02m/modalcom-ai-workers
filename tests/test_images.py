"""Tests for Modal image builders."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ai_workers.common.images import (
    PYTHON_VERSION,
    transformers_audio_image,
    transformers_image,
    vllm_image,
)


class TestImageBuilders:
    """Test Modal image builder functions."""

    @patch("modal.Image.debian_slim")
    def test_vllm_image_builder(self, mock_debian_slim: MagicMock) -> None:
        """Test vllm_image builder configuration."""
        # Setup mock chain
        mock_image = MagicMock()
        mock_debian_slim.return_value = mock_image
        mock_image.pip_install.return_value = mock_image
        mock_image.env.return_value = mock_image

        # Execute
        vllm_image()

        # Verify base image
        mock_debian_slim.assert_called_once_with(python_version=PYTHON_VERSION)

        # Verify pip packages
        mock_image.pip_install.assert_called_once()
        call_args = mock_image.pip_install.call_args[0]
        assert "vllm>=0.8" in call_args
        assert "fastapi>=0.115" in call_args
        assert "loguru>=0.7" in call_args

        # Verify env vars
        mock_image.env.assert_called_once_with({"HF_HUB_OFFLINE": "1"})

    @patch("modal.Image.debian_slim")
    def test_transformers_image_builder(self, mock_debian_slim: MagicMock) -> None:
        """Test transformers_image builder configuration."""
        # Setup mock chain
        mock_image = MagicMock()
        mock_debian_slim.return_value = mock_image
        mock_image.pip_install.return_value = mock_image
        mock_image.env.return_value = mock_image

        # Test default (no flash_attn)
        transformers_image()

        # Verify base image
        mock_debian_slim.assert_called_with(python_version=PYTHON_VERSION)

        # Verify packages
        mock_image.pip_install.assert_called()
        call_args = mock_image.pip_install.call_args[0]
        assert "torch>=2.4" in call_args
        assert "transformers>=4.47" in call_args
        # Ensure flash-attn is NOT present
        assert not any("flash-attn" in pkg for pkg in call_args)

        mock_image.env.assert_called_with({"HF_HUB_OFFLINE": "1"})

        # Reset mocks for next case
        mock_debian_slim.reset_mock()
        mock_image.reset_mock()

        # Test with flash_attn=True
        transformers_image(flash_attn=True)

        # Verify base image
        mock_debian_slim.assert_called_with(python_version=PYTHON_VERSION)

        # Verify packages
        mock_image.pip_install.assert_called()
        call_args = mock_image.pip_install.call_args[0]
        assert "torch>=2.4" in call_args
        # Ensure flash-attn IS present
        assert any("flash-attn>=2.6" in pkg for pkg in call_args)

        mock_image.env.assert_called_with({"HF_HUB_OFFLINE": "1"})

    @patch("modal.Image.debian_slim")
    def test_transformers_audio_image_builder(self, mock_debian_slim: MagicMock) -> None:
        """Test transformers_audio_image builder configuration."""
        # Setup mock chain
        mock_image = MagicMock()
        mock_debian_slim.return_value = mock_image
        mock_image.apt_install.return_value = mock_image
        mock_image.pip_install.return_value = mock_image
        mock_image.env.return_value = mock_image

        # Execute
        transformers_audio_image()

        # Verify base image
        mock_debian_slim.assert_called_once_with(python_version=PYTHON_VERSION)

        # Verify apt packages
        mock_image.apt_install.assert_called_once()
        call_args = mock_image.apt_install.call_args[0]
        assert "ffmpeg" in call_args
        assert "libsndfile1" in call_args

        # Verify pip packages
        mock_image.pip_install.assert_called_once()
        call_args = mock_image.pip_install.call_args[0]
        assert "torch>=2.4" in call_args
        assert "librosa>=0.10" in call_args
        assert "soundfile>=0.12" in call_args
        assert "python-multipart>=0.0.9" in call_args

        # Verify env vars
        mock_image.env.assert_called_once_with({"HF_HUB_OFFLINE": "1"})
