"""Tests for R2 storage helpers."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from ai_workers.common.r2 import R2Config, upload_directory

if TYPE_CHECKING:
    from pathlib import Path


class TestR2Config:
    """Test R2 configuration validation and loading."""

    def test_validate_missing_fields(self) -> None:
        """Should raise ValueError if required fields are missing."""
        # Empty config (defaults are empty strings except bucket)
        config = R2Config()
        with pytest.raises(ValueError, match="Missing R2 environment variables"):
            config.validate()

    def test_validate_success(self) -> None:
        """Should not raise if all fields are present."""
        config = R2Config(
            account_id="acc123",
            access_key_id="key123",
            secret_access_key="secret123",
        )
        config.validate()  # Should not raise

    def test_from_env(self) -> None:
        """Should load configuration from environment variables."""
        env_vars = {
            "R2_BUCKET_NAME": "test-bucket",
            "CF_ACCOUNT_ID": "env-acc",
            "CF_R2_ACCESS_KEY": "env-key",
            "CF_R2_SECRET_KEY": "env-secret",
        }
        with patch.dict(os.environ, env_vars):
            config = R2Config.from_env()
            assert config.bucket_name == "test-bucket"
            assert config.account_id == "env-acc"
            assert config.access_key_id == "env-key"
            assert config.secret_access_key == "env-secret"


class TestUploadDirectory:
    """Test directory upload functionality."""

    @pytest.fixture
    def mock_s3_client(self):
        with patch("ai_workers.common.r2.get_s3_client") as mock:
            client = MagicMock()
            mock.return_value = client
            yield client

    def test_upload_success(self, tmp_path: Path, mock_s3_client: MagicMock) -> None:
        """Should upload all files in directory recursively."""
        # Setup files
        (tmp_path / "file1.txt").write_text("content1")
        sub_dir = tmp_path / "subdir"
        sub_dir.mkdir()
        (sub_dir / "file2.txt").write_text("content2")

        config = R2Config(
            bucket_name="my-bucket",
            account_id="acc",
            access_key_id="key",
            secret_access_key="secret",
        )

        count = upload_directory(tmp_path, "prefix", config)

        assert count == 2

        # Verify calls
        # We expect 2 calls to upload_file
        assert mock_s3_client.upload_file.call_count == 2

        calls = mock_s3_client.upload_file.call_args_list
        # Extract arguments from calls
        args_list = [c.args for c in calls]

        # Expected args: (local_path_str, bucket, key)
        expected_file1 = (
            str(tmp_path / "file1.txt"),
            "my-bucket",
            "prefix/file1.txt",
        )
        expected_file2 = (
            str(sub_dir / "file2.txt"),
            "my-bucket",
            "prefix/subdir/file2.txt",
        )

        # Check both files were uploaded
        # Since implementation uses sorted(rglob), order should be deterministic but safer to check existence
        assert expected_file1 in args_list
        assert expected_file2 in args_list

    def test_upload_empty(self, tmp_path: Path, mock_s3_client: MagicMock) -> None:
        """Should handle empty directory gracefully."""
        config = R2Config(
            account_id="acc",
            access_key_id="key",
            secret_access_key="secret",
        )
        count = upload_directory(tmp_path, "prefix", config)
        assert count == 0
        mock_s3_client.upload_file.assert_not_called()

    def test_upload_skips_directories(self, tmp_path: Path, mock_s3_client: MagicMock) -> None:
        """Should only upload files, not directories."""
        # Create a directory but no files
        (tmp_path / "emptysubdir").mkdir()

        config = R2Config(
            account_id="acc",
            access_key_id="key",
            secret_access_key="secret",
        )
        count = upload_directory(tmp_path, "prefix", config)

        assert count == 0
        mock_s3_client.upload_file.assert_not_called()
