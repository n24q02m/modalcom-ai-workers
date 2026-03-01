"""Tests for cli/upload.py covering previously uncovered lines.

Covers: model=None (45-48), backup_gdrive (98-99), _sync_gdrive (104-120),
list_available (123-144), and generic exception path (93-95).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer.testing
from click.exceptions import Exit as ClickExit

from ai_workers.cli.upload import _sync_gdrive, _upload_single, app


class TestUploadCLIExtra:
    """Tests for upload CLI covering uncovered branches."""

    def _invoke(self, args: list[str]):
        runner = typer.testing.CliRunner()
        return runner.invoke(app, args)

    def test_model_none_shows_error(self) -> None:
        """Calling upload with no model arg should print error and exit non-zero (lines 45-48)."""
        result = self._invoke([])
        assert result.exit_code != 0
        assert "Cần chỉ định" in result.output or "help" in result.output.lower()

    def test_upload_all_calls_upload_single_for_each_model(self, tmp_path: Path) -> None:
        """model='all' should call _upload_single for each model (line 54-59)."""
        with patch("ai_workers.cli.upload._upload_single") as mock_single:
            result = self._invoke(["all", "--converted-dir", str(tmp_path)])
        assert mock_single.call_count > 0

    def test_upload_with_backup_gdrive_calls_sync(self, tmp_path: Path) -> None:
        """--backup-gdrive flag should trigger _sync_gdrive call (lines 98-99)."""
        model_dir = tmp_path / "qwen3-embedding-0.6b"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(b"x" * 10)

        mock_r2_config = MagicMock()
        mock_upload = MagicMock(return_value=1)

        with (
            patch("ai_workers.cli.upload.R2Config.from_env", return_value=mock_r2_config),
            patch("ai_workers.cli.upload.upload_directory", mock_upload),
            patch("ai_workers.cli.upload._sync_gdrive") as mock_sync,
        ):
            result = self._invoke(
                [
                    "--backup-gdrive",
                    "qwen3-embedding-0.6b",
                    "--converted-dir",
                    str(tmp_path),
                ]
            )
        mock_sync.assert_called_once()


class TestUploadSingleExtra:
    """Tests for _upload_single error paths."""

    def test_invalid_model_raises_exit(self, tmp_path: Path) -> None:
        """Invalid model name should raise Exit."""
        with pytest.raises(ClickExit):
            _upload_single("nonexistent-model", tmp_path)

    def test_missing_local_dir_raises_exit(self, tmp_path: Path) -> None:
        """Missing local dir should raise Exit (line 79-80)."""
        with pytest.raises(ClickExit):
            _upload_single("qwen3-embedding-0.6b", tmp_path)

    def test_upload_value_error_exits(self, tmp_path: Path) -> None:
        """ValueError from R2Config.from_env should raise Exit (lines 90-92)."""
        model_dir = tmp_path / "qwen3-embedding-0.6b"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(b"x")

        with patch("ai_workers.cli.upload.R2Config.from_env", side_effect=ValueError("No env")):
            with pytest.raises(ClickExit):
                _upload_single("qwen3-embedding-0.6b", tmp_path)

    def test_upload_generic_exception_exits(self, tmp_path: Path) -> None:
        """Generic exception from upload_directory should raise Exit (lines 93-95)."""
        model_dir = tmp_path / "qwen3-embedding-0.6b"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(b"x")

        mock_r2_config = MagicMock()
        with (
            patch("ai_workers.cli.upload.R2Config.from_env", return_value=mock_r2_config),
            patch(
                "ai_workers.cli.upload.upload_directory",
                side_effect=ConnectionError("S3 error"),
            ),
            pytest.raises(ClickExit),
        ):
            _upload_single("qwen3-embedding-0.6b", tmp_path)


class TestSyncGdrive:
    """Tests for _sync_gdrive function (lines 102-120)."""

    @patch("subprocess.run")
    def test_sync_gdrive_success(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """Successful rclone sync should not raise."""
        mock_run.return_value = MagicMock(returncode=0)
        _sync_gdrive(tmp_path, "models/test-model")
        mock_run.assert_called_once()
        cmd = mock_run.call_args.args[0]
        assert cmd[0] == "rclone"
        assert "gdrive:ai-workers-models/models/test-model" in cmd

    @patch("subprocess.run")
    def test_sync_gdrive_rclone_not_found(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """Missing rclone should print warning but NOT raise (lines 117-118)."""
        mock_run.side_effect = FileNotFoundError()
        # Should NOT raise — just warn
        _sync_gdrive(tmp_path, "models/test-model")

    @patch("subprocess.run")
    def test_sync_gdrive_called_process_error(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """rclone failure should print error but NOT raise (lines 119-120)."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "rclone sync")
        # Should NOT raise — just log error
        _sync_gdrive(tmp_path, "models/test-model")


class TestListAvailable:
    """Tests for list_available function (lines 123-144)."""

    def test_list_no_converted_dir(self) -> None:
        """list_available with no ./converted dir should show warning (lines 125-128)."""
        from ai_workers.cli.upload import list_available

        with patch("ai_workers.cli.upload.DEFAULT_CONVERTED_DIR", Path("/nonexistent/path")):
            # Should NOT raise — just print warning
            list_available()

    def test_list_with_converted_dir(self, tmp_path: Path) -> None:
        """list_available with existing dir should show table (lines 130-144)."""
        # Create fake model dirs
        model_dir = tmp_path / "qwen3-embedding-0.6b"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(b"x" * 1024)
        (model_dir / "tokenizer.json").write_bytes(b"t" * 512)

        from ai_workers.cli.upload import list_available

        with patch("ai_workers.cli.upload.DEFAULT_CONVERTED_DIR", tmp_path):
            # Should NOT raise
            list_available()
