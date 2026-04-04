"""Tests for cli/onnx_convert.py — list_onnx_models and _onnx_convert_remote."""

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from ai_workers.cli.onnx_convert import app
from ai_workers.workers.onnx_converter import ONNX_MODELS

runner = CliRunner()


# ---------------------------------------------------------------------------
# list_onnx_models
# ---------------------------------------------------------------------------


def test_list_onnx_models_output():
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    # Should print table with known model names
    for name in ONNX_MODELS:
        assert name in result.output


# ---------------------------------------------------------------------------
# _onnx_convert_remote — unknown model exits 1
# ---------------------------------------------------------------------------


def test_onnx_convert_unknown_model_exits_1():
    result = runner.invoke(app, ["not-a-real-model"])
    assert result.exit_code == 1
    assert "not-a-real-model" in result.output


# ---------------------------------------------------------------------------
# _onnx_convert_remote — no model arg exits 1
# ---------------------------------------------------------------------------


def test_onnx_convert_no_model_exits_1():
    result = runner.invoke(app, [])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# _onnx_convert_remote — success status
# ---------------------------------------------------------------------------


def test_onnx_convert_success(monkeypatch):
    mock_result = {
        "status": "success",
        "model_name": "qwen3-embedding-0.6b-onnx",
        "hf_target": "n24q02m/Qwen3-Embedding-0.6B-ONNX",
        "files_count": 5,
        "total_size_mb": 200.0,
        "url": "https://huggingface.co/n24q02m/Qwen3-Embedding-0.6B-ONNX",
        "variants": {
            "int8": {"file": "onnx/model_quantized.onnx", "size_mb": 120.0},
            "q4f16": {"file": "onnx/model_q4f16.onnx", "size_mb": 80.0},
        },
    }

    # Mock the modal app.run context manager and remote function
    mock_remote_fn = MagicMock(return_value=mock_result)
    mock_onnx_convert_model = MagicMock()
    mock_onnx_convert_model.remote = mock_remote_fn

    mock_app_run_cm = MagicMock()
    mock_app_run_cm.__enter__ = MagicMock(return_value=None)
    mock_app_run_cm.__exit__ = MagicMock(return_value=False)

    mock_onnx_convert_app = MagicMock()
    mock_onnx_convert_app.run.return_value = mock_app_run_cm

    with (
        patch("ai_workers.cli.onnx_convert.modal") as mock_modal,
        patch("ai_workers.cli.onnx_convert.onnx_convert_app", mock_onnx_convert_app),
        patch("ai_workers.cli.onnx_convert.onnx_convert_model", mock_onnx_convert_model),
    ):
        mock_modal.enable_output.return_value = mock_app_run_cm

        result = runner.invoke(app, ["qwen3-embedding-0.6b-onnx"])

    assert result.exit_code == 0
    assert "THANH CONG" in result.output or "success" in result.output.lower()


# ---------------------------------------------------------------------------
# _onnx_convert_remote — skipped status
# ---------------------------------------------------------------------------


def test_onnx_convert_skipped(monkeypatch):
    mock_result = {
        "status": "skipped",
        "model_name": "qwen3-embedding-0.6b-onnx",
        "reason": "already_exists",
        "hf_target": "n24q02m/Qwen3-Embedding-0.6B-ONNX",
    }

    mock_remote_fn = MagicMock(return_value=mock_result)
    mock_onnx_convert_model = MagicMock()
    mock_onnx_convert_model.remote = mock_remote_fn

    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=None)
    mock_cm.__exit__ = MagicMock(return_value=False)

    mock_onnx_convert_app = MagicMock()
    mock_onnx_convert_app.run.return_value = mock_cm

    with (
        patch("ai_workers.cli.onnx_convert.modal") as mock_modal,
        patch("ai_workers.cli.onnx_convert.onnx_convert_app", mock_onnx_convert_app),
        patch("ai_workers.cli.onnx_convert.onnx_convert_model", mock_onnx_convert_model),
    ):
        mock_modal.enable_output.return_value = mock_cm
        result = runner.invoke(app, ["qwen3-embedding-0.6b-onnx"])

    assert result.exit_code == 0
    assert "Bo qua" in result.output or "skipped" in result.output.lower()


# ---------------------------------------------------------------------------
# _onnx_convert_remote — AuthError exits 1
# ---------------------------------------------------------------------------


def test_onnx_convert_auth_error():
    auth_error_cls = type("AuthError", (Exception,), {})

    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(side_effect=auth_error_cls("not authenticated"))
    mock_cm.__exit__ = MagicMock(return_value=False)

    mock_onnx_convert_app = MagicMock()
    mock_onnx_convert_app.run.return_value = mock_cm

    with (
        patch("ai_workers.cli.onnx_convert.modal") as mock_modal,
        patch("ai_workers.cli.onnx_convert.onnx_convert_app", mock_onnx_convert_app),
    ):
        mock_modal.enable_output.return_value = mock_cm
        mock_modal.exception.AuthError = auth_error_cls
        result = runner.invoke(app, ["qwen3-embedding-0.6b-onnx"])

    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# onnx-convert all
# ---------------------------------------------------------------------------


def test_onnx_convert_all_success():
    mock_result = {
        "status": "success",
        "model_name": "test",
        "hf_target": "org/test",
        "files_count": 1,
        "total_size_mb": 100.0,
        "url": "https://huggingface.co/org/test",
        "variants": {},
    }

    mock_remote_fn = MagicMock(return_value=mock_result)
    mock_onnx_convert_model = MagicMock()
    mock_onnx_convert_model.remote = mock_remote_fn

    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=None)
    mock_cm.__exit__ = MagicMock(return_value=False)

    mock_onnx_convert_app = MagicMock()
    mock_onnx_convert_app.run.return_value = mock_cm

    with (
        patch("ai_workers.cli.onnx_convert.modal") as mock_modal,
        patch("ai_workers.cli.onnx_convert.onnx_convert_app", mock_onnx_convert_app),
        patch("ai_workers.cli.onnx_convert.onnx_convert_model", mock_onnx_convert_model),
    ):
        mock_modal.enable_output.return_value = mock_cm
        result = runner.invoke(app, ["all"])

    assert result.exit_code == 0
