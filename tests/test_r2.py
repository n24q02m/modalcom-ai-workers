"""Tests for R2 storage configuration and utilities.

Validates R2Config env var handling, validation logic, and S3 client creation.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from ai_workers.common.r2 import R2Config, get_s3_client, upload_directory


class TestR2Config:
    """Test R2Config dataclass."""

    def test_default_values(self) -> None:
        config = R2Config()
        assert config.bucket_name == "ai-workers-models"
        assert config.endpoint_url == ""
        assert config.access_key_id == ""
        assert config.secret_access_key == ""

    def test_custom_values(self) -> None:
        config = R2Config(
            bucket_name="my-bucket",
            endpoint_url="https://abc123.r2.cloudflarestorage.com",
            access_key_id="key123",
            secret_access_key="secret456",
        )
        assert config.bucket_name == "my-bucket"
        assert config.endpoint_url == "https://abc123.r2.cloudflarestorage.com"
        assert config.access_key_id == "key123"
        assert config.secret_access_key == "secret456"

    def test_frozen(self) -> None:
        config = R2Config()
        with pytest.raises(AttributeError):
            config.bucket_name = "changed"  # type: ignore[misc]


class TestR2ConfigFromEnv:
    """Test R2Config.from_env() reads correct environment variables."""

    def test_from_env_all_set(self) -> None:
        env = {
            "R2_BUCKET_NAME": "test-bucket",
            "R2_ENDPOINT_URL": "https://test.r2.cloudflarestorage.com",
            "R2_ACCESS_KEY_ID": "access123",
            "R2_SECRET_ACCESS_KEY": "secret789",
        }
        with patch.dict(os.environ, env, clear=False):
            config = R2Config.from_env()
        assert config.bucket_name == "test-bucket"
        assert config.endpoint_url == "https://test.r2.cloudflarestorage.com"
        assert config.access_key_id == "access123"
        assert config.secret_access_key == "secret789"

    def test_from_env_defaults(self) -> None:
        """When no env vars are set, defaults should be used."""
        with patch.dict(os.environ, {}, clear=True):
            config = R2Config.from_env()
        assert config.bucket_name == "ai-workers-models"
        assert config.endpoint_url == ""
        assert config.access_key_id == ""
        assert config.secret_access_key == ""

    def test_from_env_partial(self) -> None:
        """When only some env vars are set."""
        env = {"R2_BUCKET_NAME": "partial-bucket"}
        with patch.dict(os.environ, env, clear=True):
            config = R2Config.from_env()
        assert config.bucket_name == "partial-bucket"
        assert config.endpoint_url == ""

    def test_from_env_does_not_read_old_var_names(self) -> None:
        """Ensure old CF_ACCOUNT_ID, CF_R2_ACCESS_KEY etc. are NOT read."""
        env = {
            "CF_ACCOUNT_ID": "old-account",
            "CF_R2_ACCESS_KEY": "old-key",
            "CF_R2_SECRET_KEY": "old-secret",
        }
        with patch.dict(os.environ, env, clear=True):
            config = R2Config.from_env()
        # Old env var names should NOT populate the config
        assert config.endpoint_url == ""
        assert config.access_key_id == ""
        assert config.secret_access_key == ""


class TestR2ConfigValidate:
    """Test R2Config.validate() error handling."""

    def test_validate_all_present(self) -> None:
        config = R2Config(
            endpoint_url="https://x.r2.cloudflarestorage.com",
            access_key_id="key",
            secret_access_key="secret",
        )
        config.validate()  # Should not raise

    def test_validate_missing_endpoint(self) -> None:
        config = R2Config(access_key_id="key", secret_access_key="secret")
        with pytest.raises(ValueError, match="R2_ENDPOINT_URL"):
            config.validate()

    def test_validate_missing_access_key(self) -> None:
        config = R2Config(
            endpoint_url="https://x.r2.cloudflarestorage.com",
            secret_access_key="secret",
        )
        with pytest.raises(ValueError, match="R2_ACCESS_KEY_ID"):
            config.validate()

    def test_validate_missing_secret_key(self) -> None:
        config = R2Config(
            endpoint_url="https://x.r2.cloudflarestorage.com",
            access_key_id="key",
        )
        with pytest.raises(ValueError, match="R2_SECRET_ACCESS_KEY"):
            config.validate()

    def test_validate_all_missing(self) -> None:
        config = R2Config()
        with pytest.raises(ValueError) as exc_info:
            config.validate()
        error_msg = str(exc_info.value)
        assert "R2_ENDPOINT_URL" in error_msg
        assert "R2_ACCESS_KEY_ID" in error_msg
        assert "R2_SECRET_ACCESS_KEY" in error_msg


class TestGetS3Client:
    """Test S3 client creation."""

    def test_creates_client_with_correct_params(self) -> None:
        mock_boto3 = MagicMock()
        config = R2Config(
            endpoint_url="https://abc.r2.cloudflarestorage.com",
            access_key_id="key123",
            secret_access_key="secret456",
        )
        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            get_s3_client(config)
        mock_boto3.client.assert_called_once_with(
            "s3",
            endpoint_url="https://abc.r2.cloudflarestorage.com",
            aws_access_key_id="key123",
            aws_secret_access_key="secret456",
            region_name="auto",
        )

    def test_validates_config_before_creating_client(self) -> None:
        """get_s3_client should fail if config is invalid."""
        config = R2Config()  # All empty
        with pytest.raises(ValueError, match="Missing R2"):
            get_s3_client(config)


class TestUploadDirectory:
    """Test directory upload to R2."""

    @patch("ai_workers.common.r2.get_s3_client")
    def test_upload_empty_directory(self, mock_client: MagicMock, tmp_path) -> None:
        config = R2Config(
            endpoint_url="https://x.r2.cloudflarestorage.com",
            access_key_id="key",
            secret_access_key="secret",
        )
        count = upload_directory(tmp_path, "test-prefix", config)
        assert count == 0

    @patch("ai_workers.common.r2.get_s3_client")
    def test_upload_with_files(self, mock_client: MagicMock, tmp_path) -> None:
        # Create test files
        (tmp_path / "model.safetensors").write_bytes(b"fake model data")
        (tmp_path / "config.json").write_text('{"key": "value"}')

        mock_s3 = MagicMock()
        mock_client.return_value = mock_s3

        config = R2Config(
            endpoint_url="https://x.r2.cloudflarestorage.com",
            access_key_id="key",
            secret_access_key="secret",
        )
        count = upload_directory(tmp_path, "my-model", config)
        assert count == 2
        assert mock_s3.upload_file.call_count == 2

    @patch("ai_workers.common.r2.get_s3_client")
    def test_upload_preserves_subdirectory_structure(
        self, mock_client: MagicMock, tmp_path
    ) -> None:
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "weights.bin").write_bytes(b"data")

        mock_s3 = MagicMock()
        mock_client.return_value = mock_s3

        config = R2Config(
            endpoint_url="https://x.r2.cloudflarestorage.com",
            access_key_id="key",
            secret_access_key="secret",
        )
        upload_directory(tmp_path, "prefix", config)

        # Verify the key preserves subdirectory structure
        call_args = mock_s3.upload_file.call_args_list[0]
        key = call_args.args[2] if call_args.args else call_args[0][2]
        assert key == "prefix/subdir/weights.bin"
