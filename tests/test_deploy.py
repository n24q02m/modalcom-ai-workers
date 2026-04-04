"""Tests for deploy CLI command construction and logic.

Validates command generation, dry-run mode, and error handling
without actually calling modal deploy.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest
from click.exceptions import Exit as ClickExit
from rich.table import Table

from ai_workers.cli.deploy import (
    _deploy_module,
    _deploy_single,
    _module_to_file_path,
    deploy,
    list_workers,
)
from ai_workers.common.config import GPU, MODEL_REGISTRY, ModelConfig, ServingEngine, Task, Tier


class TestModuleToFilePath:
    """Test module path to file path conversion."""

    def test_simple_module(self) -> None:
        assert _module_to_file_path("ai_workers.workers.embedding") == (
            "src/ai_workers/workers/embedding.py"
        )


class TestDeploySingleDryRun:
    """Test deploy single model in dry-run mode."""

    @patch("ai_workers.cli.deploy.subprocess")
    def test_deploy_single_dry_run(self, mock_subprocess: MagicMock) -> None:
        """Dry-run should not call subprocess.run."""
        _deploy_single("qwen3-embedding-0.6b", dry_run=True)
        mock_subprocess.run.assert_not_called()

    @patch("ai_workers.cli.deploy.subprocess")
    def test_all_models_dry_run(self, mock_subprocess: MagicMock) -> None:
        """All registered models should support dry-run."""
        for name in MODEL_REGISTRY:
            _deploy_single(name, dry_run=True)
        mock_subprocess.run.assert_not_called()


class TestDeploySingleErrors:
    """Test deploy error handling."""

    @patch("ai_workers.cli.deploy.console")
    def test_invalid_model_name(self, mock_console: MagicMock) -> None:
        """Invalid model name should raise Exit (via typer.Exit -> click.Exit)."""
        with pytest.raises(ClickExit) as excinfo:
            _deploy_single("nonexistent-model")

        assert excinfo.value.exit_code == 1
        mock_console.print.assert_called_once()
        args, _ = mock_console.print.call_args
        assert "[red]Error:" in args[0]
        assert "nonexistent-model" in args[0]
        assert "not found" in args[0]

    @patch("ai_workers.cli.deploy.subprocess")
    def test_modal_not_found(self, mock_subprocess: MagicMock) -> None:
        """Missing modal CLI should raise Exit."""
        mock_subprocess.run.side_effect = FileNotFoundError()
        with pytest.raises(ClickExit):
            _deploy_single("qwen3-embedding-0.6b")

    @patch("ai_workers.cli.deploy.subprocess")
    def test_deploy_failure(self, mock_subprocess: MagicMock) -> None:
        """Failed deploy should raise Exit."""
        mock_subprocess.CalledProcessError = subprocess.CalledProcessError
        mock_subprocess.run.side_effect = subprocess.CalledProcessError(1, "modal deploy")
        with pytest.raises(ClickExit):
            _deploy_single("qwen3-embedding-0.6b")

    @patch("ai_workers.cli.deploy.get_model")
    @patch("ai_workers.cli.deploy.console")
    def test_deploy_single_missing_config(
        self, mock_console: MagicMock, mock_get_model: MagicMock
    ) -> None:
        """Should skip if worker_module or modal_app_var is missing."""
        # Missing worker_module
        mock_get_model.return_value = ModelConfig(
            name="m1", hf_id="h1", task=Task.EMBEDDING, tier=Tier.LIGHT, worker_module="", modal_app_var="app1"
        )
        _deploy_single("m1")
        mock_console.print.assert_any_call("[yellow]Skipping m1 -- no worker_module configured[/yellow]")

        # Missing modal_app_var
        mock_get_model.return_value = ModelConfig(
            name="m2", hf_id="h1", task=Task.EMBEDDING, tier=Tier.LIGHT, worker_module="mod1", modal_app_var=""
        )
        _deploy_single("m2")
        mock_console.print.assert_any_call("[yellow]Skipping m2 -- no modal_app_var configured[/yellow]")


class TestModalCommandConstruction:
    """Test that correct modal deploy commands are constructed."""

    @patch("ai_workers.cli.deploy.subprocess")
    def test_basic_command(self, mock_subprocess: MagicMock) -> None:
        _deploy_single("qwen3-embedding-0.6b")
        args, _ = mock_subprocess.run.call_args
        cmd = args[0]
        assert cmd[0] == "modal"
        assert cmd[1] == "deploy"
        assert "src/ai_workers/workers/embedding.py" in cmd[2]

    @patch("ai_workers.cli.deploy.subprocess")
    def test_all_models_command(self, mock_subprocess: MagicMock) -> None:
        """All registered models should generate valid deploy commands."""
        for name in MODEL_REGISTRY:
            _deploy_single(name)
        assert mock_subprocess.run.call_count == len(MODEL_REGISTRY)


class TestDeployModule:
    """Test deploy module function."""

    @patch("ai_workers.cli.deploy.subprocess")
    def test_deploy_module_dry_run(self, mock_subprocess: MagicMock) -> None:
        _deploy_module("ai_workers.workers.embedding", dry_run=True)
        mock_subprocess.run.assert_not_called()

    @patch("ai_workers.cli.deploy.subprocess")
    def test_deploy_module_success(self, mock_subprocess: MagicMock) -> None:
        _deploy_module("ai_workers.workers.embedding")
        mock_subprocess.run.assert_called_once()


class TestDeployAllGrouping:
    """Test that deploy --all correctly groups models by module."""

    def test_models_share_modules(self) -> None:
        """Light and heavy variants of same task should share a worker module."""
        modules: dict[str, list[str]] = {}
        for config in MODEL_REGISTRY.values():
            if config.worker_module:
                modules.setdefault(config.worker_module, []).append(config.name)

        # Embedding: 2 models share one module
        assert len(modules["ai_workers.workers.embedding"]) == 2
        # Reranker: 1 model
        assert len(modules["ai_workers.workers.reranker"]) == 1
        # VL Embedding: 2 models share one module
        assert len(modules["ai_workers.workers.vl_embedding"]) == 2
        # VL Reranker: 1 model
        assert len(modules["ai_workers.workers.vl_reranker"]) == 1
        # OCR: 1 model
        assert len(modules["ai_workers.workers.ocr"]) == 1
        # TTS: 2 models (light + heavy) share one module
        assert len(modules["ai_workers.workers.tts"]) == 2

    def test_total_unique_deploy_targets(self) -> None:
        """Verify the number of unique (module, app_var) pairs."""
        seen = set()
        for config in MODEL_REGISTRY.values():
            if config.worker_module and config.modal_app_var:
                seen.add((config.worker_module, config.modal_app_var))

        # Expected unique apps:
        # 1. embedding.py::app
        # 2. reranker.py::app
        # 3. vl_embedding.py::app
        # 4. vl_reranker.py::app
        # 5. ocr.py::app
        # 6. tts.py::app
        # 7. asr.py::app
        assert len(seen) == 7


class TestModelSharing:
    """Test shared attributes between models."""

    @pytest.mark.parametrize(
        "light,heavy",
        [
            ("qwen3-embedding-0.6b", "qwen3-embedding-8b"),
            ("qwen3-vl-embedding-2b", "qwen3-vl-embedding-8b"),
            ("qwen3-tts-0.6b", "qwen3-tts-1.7b"),
        ],
    )
    def test_shared_modal_app(self, light: str, heavy: str) -> None:
        """Light and heavy variants should share the same Modal app name."""
        light_cfg = MODEL_REGISTRY[light]
        heavy_cfg = MODEL_REGISTRY[heavy]

        assert light_cfg.modal_app_var == heavy_cfg.modal_app_var, (
            f"{light} and {heavy} should share modal_app_var"
        )
        assert light_cfg.modal_app_name == heavy_cfg.modal_app_name, (
            f"{light} and {heavy} should share modal_app_name"
        )


class TestWorkerModuleFilesExist:
    """Validate that worker module files actually exist on disk."""

    @pytest.mark.parametrize(
        "module",
        sorted({c.worker_module for c in MODEL_REGISTRY.values() if c.worker_module}),
    )
    def test_worker_file_exists(self, module: str) -> None:
        """Each worker module file should exist."""
        from pathlib import Path

        file_path = Path("src") / module.replace(".", "/")
        file_path = file_path.with_suffix(".py")
        assert file_path.exists(), f"Worker file not found: {file_path}"


class TestDeployCoverage:
    """Extended coverage for deploy CLI command."""

    def test_deploy_no_worker_error(self) -> None:
        """Calling deploy without worker or --all should raise Exit."""
        with patch("ai_workers.cli.deploy.console") as mock_console:
            with pytest.raises(ClickExit) as excinfo:
                deploy(worker=None, all_workers=False, dry_run=False)
            assert excinfo.value.exit_code == 1
            mock_console.print.assert_called_with("[red]Error: specify a model name or use --all[/red]")

    @patch("ai_workers.cli.deploy.list_models")
    @patch("ai_workers.cli.deploy._deploy_app")
    @patch("ai_workers.cli.deploy.console")
    def test_deploy_all_success(
        self, mock_console: MagicMock, mock_deploy_app: MagicMock, mock_list_models: MagicMock
    ) -> None:
        """deploy --all should call _deploy_app for each unique module/app_var."""
        mock_list_models.return_value = [
            ModelConfig(
                name="m1",
                hf_id="h1",
                task=Task.EMBEDDING,
                tier=Tier.LIGHT,
                worker_module="mod1",
                modal_app_var="app1",
                modal_app_name="n1",
                gpu=GPU.T4,
                serving_engine=ServingEngine.CUSTOM_FASTAPI,
            ),
            ModelConfig(
                name="m2",
                hf_id="h2",
                task=Task.EMBEDDING,
                tier=Tier.HEAVY,
                worker_module="mod1",
                modal_app_var="app1",
                modal_app_name="n1",
                gpu=GPU.T4,
                serving_engine=ServingEngine.CUSTOM_FASTAPI,
            ),
            ModelConfig(
                name="m3",
                hf_id="h3",
                task=Task.RERANKER_LLM,
                tier=Tier.HEAVY,
                worker_module="mod2",
                modal_app_var="app2",
                modal_app_name="n2",
                gpu=GPU.A10G,
                serving_engine=ServingEngine.VLLM,
            ),
        ]
        deploy(worker=None, all_workers=True, dry_run=True)
        assert mock_deploy_app.call_count == 2
        mock_deploy_app.assert_any_call("mod1", "app1", dry_run=True)
        mock_deploy_app.assert_any_call("mod2", "app2", dry_run=True)

    @patch("ai_workers.cli.deploy.list_models")
    @patch("ai_workers.cli.deploy._deploy_app")
    @patch("ai_workers.cli.deploy.console")
    def test_deploy_all_failure(
        self, mock_console: MagicMock, mock_deploy_app: MagicMock, mock_list_models: MagicMock
    ) -> None:
        """deploy --all should report failures and exit with 1."""
        mock_list_models.return_value = [
            ModelConfig(
                name="m1",
                hf_id="h1",
                task=Task.EMBEDDING,
                tier=Tier.LIGHT,
                worker_module="mod1",
                modal_app_var="app1",
                modal_app_name="n1",
                gpu=GPU.T4,
                serving_engine=ServingEngine.CUSTOM_FASTAPI,
            ),
        ]
        mock_deploy_app.side_effect = ClickExit(1)
        with pytest.raises(ClickExit) as excinfo:
            deploy(worker=None, all_workers=True, dry_run=False)
        assert excinfo.value.exit_code == 1
        # Check that it printed the failure
        mock_console.print.assert_any_call("\n[red bold]Deploy FAILED: m1[/red bold]")

    @patch("ai_workers.cli.deploy.MODEL_REGISTRY")
    @patch("ai_workers.cli.deploy.console")
    def test_list_workers(self, mock_console: MagicMock, mock_registry: MagicMock) -> None:
        """list_workers should print a table of models."""
        mock_registry.__iter__.return_value = ["m1"]
        mock_registry.__getitem__.return_value = ModelConfig(
            name="m1",
            hf_id="h1",
            task=Task.EMBEDDING,
            tier=Tier.LIGHT,
            worker_module="mod1",
            modal_app_var="app1",
            modal_app_name="n1",
            gpu=GPU.T4,
            serving_engine=ServingEngine.CUSTOM_FASTAPI,
        )
        list_workers()
        # verify that a table was printed
        table_called = False
        for call in mock_console.print.call_args_list:
            if isinstance(call.args[0], Table):
                table_called = True
                break
        assert table_called
