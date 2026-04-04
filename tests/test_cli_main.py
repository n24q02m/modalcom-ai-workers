"""Tests for cli/__main__.py — root CLI app structure."""


from typer.testing import CliRunner

from ai_workers.cli.__main__ import app

runner = CliRunner()


def test_cli_app_help():
    """Root app shows help output."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ai-workers" in result.output or "AI Workers" in result.output


def test_cli_app_has_onnx_convert_subcommand():
    result = runner.invoke(app, ["onnx-convert", "--help"])
    assert result.exit_code == 0


def test_cli_app_has_gguf_convert_subcommand():
    result = runner.invoke(app, ["gguf-convert", "--help"])
    assert result.exit_code == 0


def test_cli_app_has_deploy_subcommand():
    result = runner.invoke(app, ["deploy", "--help"])
    assert result.exit_code == 0
