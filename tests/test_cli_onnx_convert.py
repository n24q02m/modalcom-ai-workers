"""Tests for cli/onnx_convert.py — list_onnx_models and _onnx_convert_remote."""

from __future__ import annotations

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
# onnx_convert (callback)
# ---------------------------------------------------------------------------


def test_onnx_convert_no_model_arg():
    # Covers lines 43-46
    # Note: If no_args_is_help=True is set on the Typer object,
    # it might exit with 0 or 2 before reaching our code.
    # We try to trigger the model is None check by passing an option.
    result = runner.invoke(app, ["--force"])
    if result.exit_code == 2:
        # If still 2, it might be Typer's help/usage exit.
        # We'll try just [] and check if it hits our line.
        result = runner.invoke(app, [])

    # If it's still not hitting our code, it's likely due to no_args_is_help=True.
    # But let's see what happens.
    assert "Please specify a model" in result.output
    assert result.exit_code == 1


def test_onnx_convert_unknown_model_exits_1():
    result = runner.invoke(app, ["not-a-real-model"])
    assert result.exit_code == 1
    assert "not-a-real-model" in result.output


# ---------------------------------------------------------------------------
# _onnx_convert_remote — success status
# ---------------------------------------------------------------------------


def test_onnx_convert_success():
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

    auth_error_cls = type("AuthError", (Exception,), {})

    with (
        patch("ai_workers.cli.onnx_convert.modal") as mock_modal,
        patch("ai_workers.cli.onnx_convert.onnx_convert_app", mock_onnx_convert_app),
        patch("ai_workers.cli.onnx_convert.onnx_convert_model", mock_onnx_convert_model),
    ):
        mock_modal.exception.AuthError = auth_error_cls
        mock_modal.enable_output.return_value = mock_app_run_cm

        result = runner.invoke(app, ["qwen3-embedding-0.6b-onnx"])

    assert result.exit_code == 0
    assert "SUCCESS" in result.output


# ---------------------------------------------------------------------------
# _onnx_convert_remote — skipped status
# ---------------------------------------------------------------------------


def test_onnx_convert_skipped():
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

    auth_error_cls = type("AuthError", (Exception,), {})

    with (
        patch("ai_workers.cli.onnx_convert.modal") as mock_modal,
        patch("ai_workers.cli.onnx_convert.onnx_convert_app", mock_onnx_convert_app),
        patch("ai_workers.cli.onnx_convert.onnx_convert_model", mock_onnx_convert_model),
    ):
        mock_modal.exception.AuthError = auth_error_cls
        mock_modal.enable_output.return_value = mock_cm
        result = runner.invoke(app, ["qwen3-embedding-0.6b-onnx"])

    assert result.exit_code == 0
    assert "Skipped" in result.output


# ---------------------------------------------------------------------------
# _onnx_convert_remote — unknown status
# ---------------------------------------------------------------------------


def test_onnx_convert_unknown_status():
    # Covers lines 130-131
    mock_result = {"status": "what_is_this"}

    mock_remote_fn = MagicMock(return_value=mock_result)
    mock_onnx_convert_model = MagicMock()
    mock_onnx_convert_model.remote = mock_remote_fn

    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=None)
    mock_cm.__exit__ = MagicMock(return_value=False)

    mock_onnx_convert_app = MagicMock()
    mock_onnx_convert_app.run.return_value = mock_cm

    auth_error_cls = type("AuthError", (Exception,), {})

    with (
        patch("ai_workers.cli.onnx_convert.modal") as mock_modal,
        patch("ai_workers.cli.onnx_convert.onnx_convert_app", mock_onnx_convert_app),
        patch("ai_workers.cli.onnx_convert.onnx_convert_model", mock_onnx_convert_model),
    ):
        mock_modal.exception.AuthError = auth_error_cls
        mock_modal.enable_output.return_value = mock_cm
        result = runner.invoke(app, ["qwen3-embedding-0.6b-onnx"])

    assert result.exit_code == 1
    assert "unknown status" in result.output


# ---------------------------------------------------------------------------
# _onnx_convert_remote — AuthError
# ---------------------------------------------------------------------------


def test_onnx_convert_auth_error_from_remote():
    # Specifically covers AuthError thrown by remote() as requested
    auth_error_cls = type("AuthError", (Exception,), {})

    mock_remote_fn = MagicMock(side_effect=auth_error_cls("not authenticated"))
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
        mock_modal.exception.AuthError = auth_error_cls
        result = runner.invoke(app, ["qwen3-embedding-0.6b-onnx"])

    assert result.exit_code == 1
    assert "Modal not authenticated" in result.output


# ---------------------------------------------------------------------------
# _onnx_convert_remote — general exception
# ---------------------------------------------------------------------------


def test_onnx_convert_general_exception():
    # Covers lines 136-141
    mock_remote_fn = MagicMock(side_effect=ValueError("something went wrong"))
    mock_onnx_convert_model = MagicMock()
    mock_onnx_convert_model.remote = mock_remote_fn

    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=None)
    mock_cm.__exit__ = MagicMock(return_value=False)

    mock_onnx_convert_app = MagicMock()
    mock_onnx_convert_app.run.return_value = mock_cm

    auth_error_cls = type("AuthError", (Exception,), {})

    with (
        patch("ai_workers.cli.onnx_convert.modal") as mock_modal,
        patch("ai_workers.cli.onnx_convert.onnx_convert_app", mock_onnx_convert_app),
        patch("ai_workers.cli.onnx_convert.onnx_convert_model", mock_onnx_convert_model),
    ):
        mock_modal.exception.AuthError = auth_error_cls
        mock_modal.enable_output.return_value = mock_cm
        result = runner.invoke(app, ["qwen3-embedding-0.6b-onnx"])

    assert result.exit_code == 1
    assert "FAILED — something went wrong" in result.output


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

    auth_error_cls = type("AuthError", (Exception,), {})

    with (
        patch("ai_workers.cli.onnx_convert.modal") as mock_modal,
        patch("ai_workers.cli.onnx_convert.onnx_convert_app", mock_onnx_convert_app),
        patch("ai_workers.cli.onnx_convert.onnx_convert_model", mock_onnx_convert_model),
        patch("ai_workers.cli.onnx_convert.ONNX_MODELS", {"test": MagicMock(name="test")}),
    ):
        mock_modal.exception.AuthError = auth_error_cls
        mock_modal.enable_output.return_value = mock_cm
        result = runner.invoke(app, ["all"])

    assert result.exit_code == 0
    assert "All 1 models converted successfully" in result.output


def test_onnx_convert_all_failure():
    # Covers lines 60-61, 63-66
    mock_remote_fn = MagicMock(side_effect=Exception("boom"))
    mock_onnx_convert_model = MagicMock()
    mock_onnx_convert_model.remote = mock_remote_fn

    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=None)
    mock_cm.__exit__ = MagicMock(return_value=False)

    mock_onnx_convert_app = MagicMock()
    mock_onnx_convert_app.run.return_value = mock_cm

    auth_error_cls = type("AuthError", (Exception,), {})

    with (
        patch("ai_workers.cli.onnx_convert.modal") as mock_modal,
        patch("ai_workers.cli.onnx_convert.onnx_convert_app", mock_onnx_convert_app),
        patch("ai_workers.cli.onnx_convert.onnx_convert_model", mock_onnx_convert_model),
        patch("ai_workers.cli.onnx_convert.ONNX_MODELS", {"test": MagicMock(name="test")}),
    ):
        mock_modal.exception.AuthError = auth_error_cls
        mock_modal.enable_output.return_value = mock_cm
        result = runner.invoke(app, ["all"])

    assert result.exit_code == 1
    assert "1 model(s) failed: test" in result.output


# ---------------------------------------------------------------------------
# dry-run skips remote call
# ---------------------------------------------------------------------------


def test_onnx_convert_dry_run():
    with patch("ai_workers.cli.onnx_convert.onnx_convert_model") as mock_model:
        result = runner.invoke(app, ["--dry-run", "qwen3-embedding-0.6b-onnx"])
        assert result.exit_code == 0
        assert "dry run -- skipped" in result.output
        mock_model.remote.assert_not_called()


def test_onnx_convert_all_system_exit():
    # Covers lines 60-61 (SystemExit specifically)
    def side_effect(name, **kwargs):
        if name == "fail":
            raise SystemExit(1)
        return None

    mock_remote = MagicMock(side_effect=side_effect)

    with (
        patch("ai_workers.cli.onnx_convert._onnx_convert_remote", mock_remote),
        patch(
            "ai_workers.cli.onnx_convert.ONNX_MODELS", {"fail": MagicMock(), "pass": MagicMock()}
        ),
    ):
        result = runner.invoke(app, ["all"])

    assert result.exit_code == 1
    assert "1 model(s) failed: fail" in result.output
    assert mock_remote.call_count == 2
