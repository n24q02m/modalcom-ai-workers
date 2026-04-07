"""Tests for cli/gguf_convert.py — list_gguf_models and _gguf_convert_remote."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from ai_workers.cli.gguf_convert import app
from ai_workers.workers.gguf_converter import GGUF_MODELS

runner = CliRunner()


# ---------------------------------------------------------------------------
# list_gguf_models
# ---------------------------------------------------------------------------


def test_list_gguf_models_output():
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    for name in GGUF_MODELS:
        assert name in result.output


# ---------------------------------------------------------------------------
# unknown model exits 1
# ---------------------------------------------------------------------------


def test_gguf_convert_unknown_model_exits_1():
    result = runner.invoke(app, ["not-a-real-model"])
    assert result.exit_code == 1
    assert "not-a-real-model" in result.output


# ---------------------------------------------------------------------------
# no model arg exits 1
# ---------------------------------------------------------------------------


def test_gguf_convert_no_model_exits_1():
    result = runner.invoke(app, [])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# success status
# ---------------------------------------------------------------------------


def test_gguf_convert_success():
    mock_result = {
        "status": "success",
        "model_name": "qwen3-embedding-0.6b-gguf",
        "hf_target": "n24q02m/Qwen3-Embedding-0.6B-GGUF",
        "gguf_file": "qwen3-embedding-0.6b-q4-k-m.gguf",
        "quant_type": "Q4_K_M",
        "size_mb": 300.0,
        "url": "https://huggingface.co/n24q02m/Qwen3-Embedding-0.6B-GGUF",
    }

    mock_remote_fn = MagicMock(return_value=mock_result)
    mock_gguf_convert_model = MagicMock()
    mock_gguf_convert_model.remote = mock_remote_fn

    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=None)
    mock_cm.__exit__ = MagicMock(return_value=False)

    mock_gguf_convert_app = MagicMock()
    mock_gguf_convert_app.run.return_value = mock_cm

    with (
        patch("ai_workers.cli.gguf_convert.modal") as mock_modal,
        patch("ai_workers.cli.gguf_convert.gguf_convert_app", mock_gguf_convert_app),
        patch("ai_workers.cli.gguf_convert.gguf_convert_model", mock_gguf_convert_model),
    ):
        mock_modal.enable_output.return_value = mock_cm
        result = runner.invoke(app, ["qwen3-embedding-0.6b-gguf"])

    assert result.exit_code == 0
    assert "THANH CONG" in result.output or "success" in result.output.lower()


# ---------------------------------------------------------------------------
# skipped status
# ---------------------------------------------------------------------------


def test_gguf_convert_skipped():
    mock_result = {
        "status": "skipped",
        "model_name": "qwen3-embedding-0.6b-gguf",
        "reason": "already_exists",
        "hf_target": "n24q02m/Qwen3-Embedding-0.6B-GGUF",
    }

    mock_remote_fn = MagicMock(return_value=mock_result)
    mock_gguf_convert_model = MagicMock()
    mock_gguf_convert_model.remote = mock_remote_fn

    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=None)
    mock_cm.__exit__ = MagicMock(return_value=False)

    mock_gguf_convert_app = MagicMock()
    mock_gguf_convert_app.run.return_value = mock_cm

    with (
        patch("ai_workers.cli.gguf_convert.modal") as mock_modal,
        patch("ai_workers.cli.gguf_convert.gguf_convert_app", mock_gguf_convert_app),
        patch("ai_workers.cli.gguf_convert.gguf_convert_model", mock_gguf_convert_model),
    ):
        mock_modal.enable_output.return_value = mock_cm
        result = runner.invoke(app, ["qwen3-embedding-0.6b-gguf"])

    assert result.exit_code == 0
    assert "Bo qua" in result.output or "skipped" in result.output.lower()


# ---------------------------------------------------------------------------
# AuthError exits 1
# ---------------------------------------------------------------------------


def test_gguf_convert_auth_error():
    auth_error_cls = type("AuthError", (Exception,), {})

    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(side_effect=auth_error_cls("no auth"))
    mock_cm.__exit__ = MagicMock(return_value=False)

    mock_gguf_convert_app = MagicMock()
    mock_gguf_convert_app.run.return_value = mock_cm

    with (
        patch("ai_workers.cli.gguf_convert.modal") as mock_modal,
        patch("ai_workers.cli.gguf_convert.gguf_convert_app", mock_gguf_convert_app),
    ):
        mock_modal.enable_output.return_value = mock_cm
        mock_modal.exception.AuthError = auth_error_cls
        result = runner.invoke(app, ["qwen3-embedding-0.6b-gguf"])

    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# gguf-convert all
# ---------------------------------------------------------------------------


def test_gguf_convert_all_success():
    mock_result = {
        "status": "success",
        "model_name": "test",
        "hf_target": "org/test-GGUF",
        "gguf_file": "test-q4-k-m.gguf",
        "quant_type": "Q4_K_M",
        "size_mb": 100.0,
        "url": "https://huggingface.co/org/test-GGUF",
    }

    mock_remote_fn = MagicMock(return_value=mock_result)
    mock_gguf_convert_model = MagicMock()
    mock_gguf_convert_model.remote = mock_remote_fn

    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=None)
    mock_cm.__exit__ = MagicMock(return_value=False)

    mock_gguf_convert_app = MagicMock()
    mock_gguf_convert_app.run.return_value = mock_cm

    with (
        patch("ai_workers.cli.gguf_convert.modal") as mock_modal,
        patch("ai_workers.cli.gguf_convert.gguf_convert_app", mock_gguf_convert_app),
        patch("ai_workers.cli.gguf_convert.gguf_convert_model", mock_gguf_convert_model),
    ):
        mock_modal.enable_output.return_value = mock_cm
        result = runner.invoke(app, ["all"])

    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# gguf-convert all error handling
# ---------------------------------------------------------------------------


def test_gguf_convert_all_error_handling():
    # Use a small dict to avoid iterating over all real models
    mock_models = {"model1": MagicMock(), "model2": MagicMock()}

    with (
        patch("ai_workers.cli.gguf_convert.GGUF_MODELS", mock_models),
        patch("ai_workers.cli.gguf_convert._gguf_convert_remote") as mock_remote,
    ):

        def side_effect(name, **kwargs):
            if name == "model1":
                raise SystemExit(1)
            raise Exception("Failure")

        mock_remote.side_effect = side_effect
        result = runner.invoke(app, ["all"])

    assert result.exit_code == 1
    assert "Conversion FAILED" in result.output
    assert "model1" in result.output
    assert "model2" in result.output


# dry-run skips remote call
# ---------------------------------------------------------------------------


def test_gguf_convert_dry_run():
    with patch("ai_workers.cli.gguf_convert.gguf_convert_model") as mock_model:
        result = runner.invoke(app, ["--dry-run", "qwen3-embedding-0.6b-gguf"])
        assert result.exit_code == 0
        assert "dry run -- skipped" in result.output
        mock_model.remote.assert_not_called()


# ---------------------------------------------------------------------------
# model is None with option
# ---------------------------------------------------------------------------


def test_gguf_convert_no_model_with_option_exits_1():
    result = runner.invoke(app, ["--dry-run"])
    assert result.exit_code == 1
    assert "Please specify a model" in result.output


# ---------------------------------------------------------------------------
# unknown status
# ---------------------------------------------------------------------------


def test_gguf_convert_unknown_status():
    mock_result = {"status": "unknown"}

    mock_remote_fn = MagicMock(return_value=mock_result)
    mock_gguf_convert_model = MagicMock()
    mock_gguf_convert_model.remote = mock_remote_fn

    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=None)
    mock_cm.__exit__ = MagicMock(return_value=False)

    mock_gguf_convert_app = MagicMock()
    mock_gguf_convert_app.run.return_value = mock_cm

    with (
        patch("ai_workers.cli.gguf_convert.modal") as mock_modal,
        patch("ai_workers.cli.gguf_convert.gguf_convert_app", mock_gguf_convert_app),
        patch("ai_workers.cli.gguf_convert.gguf_convert_model", mock_gguf_convert_model),
    ):
        mock_modal.exception.AuthError = type("AuthError", (Exception,), {})
        mock_modal.enable_output.return_value = mock_cm
        result = runner.invoke(app, ["qwen3-embedding-0.6b-gguf"])

    assert result.exit_code == 1
    assert "unknown status" in result.output


# ---------------------------------------------------------------------------
# general Exception
# ---------------------------------------------------------------------------


def test_gguf_convert_general_exception():
    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(side_effect=Exception("kaboom"))
    mock_cm.__exit__ = MagicMock(return_value=False)

    mock_gguf_convert_app = MagicMock()
    mock_gguf_convert_app.run.return_value = mock_cm

    with (
        patch("ai_workers.cli.gguf_convert.modal") as mock_modal,
        patch("ai_workers.cli.gguf_convert.gguf_convert_app", mock_gguf_convert_app),
    ):
        mock_modal.exception.AuthError = type("AuthError", (Exception,), {})
        mock_modal.enable_output.return_value = mock_cm
        result = runner.invoke(app, ["qwen3-embedding-0.6b-gguf"])

    assert result.exit_code == 1
    assert "FAILED — kaboom" in result.output


# ---------------------------------------------------------------------------
# gguf-convert all loop error tracking (explicit coverage for lines 70-71)
# ---------------------------------------------------------------------------


def test_gguf_convert_all_loop_error_tracking():
    mock_models = {"model1": MagicMock(), "model2": MagicMock()}

    with (
        patch("ai_workers.cli.gguf_convert.GGUF_MODELS", mock_models),
        patch("ai_workers.cli.gguf_convert._gguf_convert_remote") as mock_remote,
    ):

        def side_effect(name, **kwargs):
            if name == "model1":
                raise SystemExit(1)
            raise Exception("Failure")

        mock_remote.side_effect = side_effect
        result = runner.invoke(app, ["all"])

    assert result.exit_code == 1
    assert "Conversion FAILED" in result.output
    assert "model1" in result.output
    assert "model2" in result.output
