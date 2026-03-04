"""Tests for cli/upload.py — upload single, upload all, list_available."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from ai_workers.cli.upload import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# upload — unknown model exits 1
# ---------------------------------------------------------------------------


def test_upload_unknown_model_exits_1():
    result = runner.invoke(app, ["not-a-real-model"])
    assert result.exit_code == 1
    assert "not-a-real-model" in result.output or "Error" in result.output


# ---------------------------------------------------------------------------
# upload — missing local_dir exits 1
# ---------------------------------------------------------------------------


def test_upload_missing_local_dir_exits_1():
    # qwen3-embedding-0.6b exists in registry but converted dir won't exist
    with tempfile.TemporaryDirectory() as tmpdir:
        result = runner.invoke(app, ["qwen3-embedding-0.6b", "--converted-dir", tmpdir])
    assert result.exit_code == 1
    assert "does not exist" in result.output


# ---------------------------------------------------------------------------
# upload — success with mock R2Config and upload_directory
# ---------------------------------------------------------------------------


def test_upload_single_success():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create the model dir so it exists
        converted_dir = Path(tmpdir)
        model_dir = converted_dir / "qwen3-embedding-0.6b"
        model_dir.mkdir()
        (model_dir / "model.bin").write_bytes(b"fake model data")

        mock_r2_config = MagicMock()
        mock_upload_dir = MagicMock(return_value=3)

        with (
            patch("ai_workers.cli.upload.R2Config") as mock_r2_config_cls,
            patch("ai_workers.cli.upload.upload_directory", mock_upload_dir),
        ):
            mock_r2_config_cls.from_env.return_value = mock_r2_config
            result = runner.invoke(
                app, ["qwen3-embedding-0.6b", "--converted-dir", str(converted_dir)]
            )

    assert result.exit_code == 0
    assert "SUCCESS" in result.output or "success" in result.output.lower()
    mock_upload_dir.assert_called_once()


# ---------------------------------------------------------------------------
# upload — R2Config ValueError exits 1
# ---------------------------------------------------------------------------


def test_upload_r2_config_missing_exits_1():
    with tempfile.TemporaryDirectory() as tmpdir:
        converted_dir = Path(tmpdir)
        model_dir = converted_dir / "qwen3-embedding-0.6b"
        model_dir.mkdir()

        with patch("ai_workers.cli.upload.R2Config") as mock_r2_config_cls:
            mock_r2_config_cls.from_env.side_effect = ValueError("R2_BUCKET not set")
            result = runner.invoke(
                app, ["qwen3-embedding-0.6b", "--converted-dir", str(converted_dir)]
            )

    assert result.exit_code == 1
    assert "R2_BUCKET" in result.output or "Error" in result.output


# ---------------------------------------------------------------------------
# upload all
# ---------------------------------------------------------------------------


def test_upload_all():
    from ai_workers.common.config import list_models

    models = list_models()
    with tempfile.TemporaryDirectory() as tmpdir:
        converted_dir = Path(tmpdir)
        # Create dirs for all models
        for m in models:
            model_dir = converted_dir / m.name
            model_dir.mkdir()
            (model_dir / "model.bin").write_bytes(b"data")

        mock_r2_config = MagicMock()
        mock_upload_dir = MagicMock(return_value=1)

        with (
            patch("ai_workers.cli.upload.R2Config") as mock_r2_config_cls,
            patch("ai_workers.cli.upload.upload_directory", mock_upload_dir),
        ):
            mock_r2_config_cls.from_env.return_value = mock_r2_config
            result = runner.invoke(app, ["all", "--converted-dir", str(converted_dir)])

    assert result.exit_code == 0
    assert mock_upload_dir.call_count == len(models)


# ---------------------------------------------------------------------------
# list_available — no converted dir
# ---------------------------------------------------------------------------


def test_list_available_no_converted_dir():
    with patch("ai_workers.cli.upload.DEFAULT_CONVERTED_DIR", Path("/nonexistent_dir_xyz")):
        result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "No converted" in result.output


# ---------------------------------------------------------------------------
# list_available — with model dirs
# ---------------------------------------------------------------------------


def test_list_available_with_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        converted_dir = Path(tmpdir)
        model_dir = converted_dir / "my-model"
        model_dir.mkdir()
        (model_dir / "model.bin").write_bytes(b"x" * 1024)

        with patch("ai_workers.cli.upload.DEFAULT_CONVERTED_DIR", converted_dir):
            result = runner.invoke(app, ["list"])

    assert result.exit_code == 0
    assert "my-model" in result.output
