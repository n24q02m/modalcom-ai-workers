"""Tests for deploy CLI command construction and logic.

Validates command generation, dry-run mode, and error handling
without actually calling modal deploy.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.exceptions import Exit as ClickExit

from ai_workers.cli.deploy import _deploy_module, _deploy_single, _module_to_file_path
from ai_workers.common.config import MODEL_REGISTRY


class TestModuleToFilePath:
    """Test module path to file path conversion."""

    def test_simple_module(self) -> None:
        assert _module_to_file_path("ai_workers.workers.embedding") == (
            "src/ai_workers/workers/embedding.py"
        )

    def test_deeply_nested(self) -> None:
        assert _module_to_file_path("a.b.c.d") == "src/a/b/c/d.py"

    def test_single_module(self) -> None:
        assert _module_to_file_path("module") == "src/module.py"


class TestDeploySingleDryRun:
    """Test deploy single model in dry-run mode."""

    @patch("ai_workers.cli.deploy.subprocess")
    def test_dry_run_does_not_call_subprocess(self, mock_subprocess: MagicMock) -> None:
        """Dry run should NOT execute any subprocess commands."""
        _deploy_single("qwen3-embedding-0.6b", dry_run=True)
        mock_subprocess.run.assert_not_called()

    @patch("ai_workers.cli.deploy.subprocess")
    def test_dry_run_all_models(self, mock_subprocess: MagicMock) -> None:
        """Dry run should work for all registered models."""
        for name in MODEL_REGISTRY:
            _deploy_single(name, dry_run=True)
        mock_subprocess.run.assert_not_called()


class TestDeploySingleErrors:
    """Test deploy error handling."""

    def test_invalid_model_name(self) -> None:
        """Invalid model name should raise Exit (via typer.Exit -> click.Exit)."""
        with pytest.raises(ClickExit):
            _deploy_single("nonexistent-model")

    @patch("ai_workers.cli.deploy.subprocess")
    def test_modal_not_found(self, mock_subprocess: MagicMock) -> None:
        """Missing modal CLI should raise Exit."""
        mock_subprocess.run.side_effect = FileNotFoundError()
        with pytest.raises(ClickExit):
            _deploy_single("qwen3-embedding-0.6b")

    @patch("ai_workers.cli.deploy.subprocess")
    def test_deploy_failure(self, mock_subprocess: MagicMock) -> None:
        """Failed deploy should raise Exit."""
        import subprocess

        mock_subprocess.run.side_effect = subprocess.CalledProcessError(1, "modal deploy")
        mock_subprocess.CalledProcessError = subprocess.CalledProcessError
        with pytest.raises(ClickExit):
            _deploy_single("qwen3-embedding-0.6b")


class TestDeployCommandConstruction:
    """Test that correct modal deploy commands are constructed."""

    @patch("ai_workers.cli.deploy.subprocess")
    def test_command_uses_file_path_and_app_var(self, mock_subprocess: MagicMock) -> None:
        """Deploy command should use file.py::app_var format."""
        mock_subprocess.run.return_value = MagicMock(returncode=0)
        _deploy_single("qwen3-embedding-0.6b")
        call_args = mock_subprocess.run.call_args
        cmd = call_args.args[0] if call_args.args else call_args[0][0]
        assert cmd[0] == "modal"
        assert cmd[1] == "deploy"
        assert cmd[2] == "src/ai_workers/workers/embedding.py::embedding_app"

    @patch("ai_workers.cli.deploy.subprocess")
    def test_all_models_generate_valid_commands(self, mock_subprocess: MagicMock) -> None:
        """All registered models should generate valid deploy commands."""
        mock_subprocess.run.return_value = MagicMock(returncode=0)

        for name, config in MODEL_REGISTRY.items():
            if not config.worker_module or not config.modal_app_var:
                continue
            _deploy_single(name)
            call_args = mock_subprocess.run.call_args
            cmd = call_args.args[0] if call_args.args else call_args[0][0]
            assert cmd[0] == "modal"
            assert "::" in cmd[2], f"{name}: command should use file.py::app_var format"
            file_part = cmd[2].split("::")[0]
            assert file_part.endswith(".py"), f"{name}: file part should end with .py"
            assert file_part.startswith("src/"), f"{name}: path should start with src/"


class TestDeployModuleDryRun:
    """Test deploy module function."""

    @patch("ai_workers.cli.deploy.subprocess")
    def test_deploy_module_dry_run(self, mock_subprocess: MagicMock) -> None:
        _deploy_module("ai_workers.workers.embedding", dry_run=True)
        mock_subprocess.run.assert_not_called()

    @patch("ai_workers.cli.deploy.subprocess")
    def test_deploy_module_success(self, mock_subprocess: MagicMock) -> None:
        mock_subprocess.run.return_value = MagicMock(returncode=0)
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

        # Embedding: 2 models (light + heavy) share one module
        assert len(modules["ai_workers.workers.embedding"]) == 2
        # Reranker: 1 model (8B only)
        assert len(modules["ai_workers.workers.reranker"]) == 1
        # VL Embedding: 2 models share one module
        assert len(modules["ai_workers.workers.vl_embedding"]) == 2
        # VL Reranker: 1 model (8B only)
        assert len(modules["ai_workers.workers.vl_reranker"]) == 1
        # OCR: 1 model
        assert len(modules["ai_workers.workers.ocr"]) == 1
        # TTS: 2 models (light + heavy) share one module
        assert len(modules["ai_workers.workers.tts"]) == 2
        # ASR: 2 models (light + heavy) share one module
        assert len(modules["ai_workers.workers.asr"]) == 2

    def test_total_unique_modules(self) -> None:
        """Should have 8 unique worker modules for 12 models."""
        modules = {c.worker_module for c in MODEL_REGISTRY.values() if c.worker_module}
        assert len(modules) == 8

    def test_total_unique_deploy_targets(self) -> None:
        """Merged apps: should have 8 unique (module, app_var) pairs for 12 models.

        Embedding, VL Embedding, TTS, ASR each merge light+heavy into one app.
        Reranker and VL Reranker have single 8B model each. Plus OCR and MM Reranker = 8 total.
        """
        targets = {
            (c.worker_module, c.modal_app_var)
            for c in MODEL_REGISTRY.values()
            if c.worker_module and c.modal_app_var
        }
        assert len(targets) == 8

    def test_merged_apps_share_app_var(self) -> None:
        """Light and heavy variants of merged tasks should share the same modal_app_var."""
        from ai_workers.common.config import get_model

        pairs = [
            ("qwen3-embedding-0.6b", "qwen3-embedding-8b"),
            ("qwen3-vl-embedding-2b", "qwen3-vl-embedding-8b"),
            ("qwen3-tts-0.6b", "qwen3-tts-1.7b"),
            ("qwen3-asr-0.6b", "qwen3-asr-1.7b"),
        ]
        for light, heavy in pairs:
            light_cfg = get_model(light)
            heavy_cfg = get_model(heavy)
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


class TestGroupDeployTargets:
    """Test grouping logic for deployment."""

    def test_group_deploy_targets_basic(self) -> None:
        """Should group models by (module, app_var)."""
        from ai_workers.cli.deploy import _group_deploy_targets
        from ai_workers.common.config import ModelConfig, Task, Tier

        models = [
            ModelConfig(
                name="m1",
                hf_id="h1",
                task=Task.EMBEDDING,
                tier=Tier.LIGHT,
                worker_module="mod1",
                modal_app_var="app1",
            ),
            ModelConfig(
                name="m2",
                hf_id="h2",
                task=Task.EMBEDDING,
                tier=Tier.HEAVY,
                worker_module="mod1",
                modal_app_var="app1",
            ),
            ModelConfig(
                name="m3",
                hf_id="h3",
                task=Task.RERANKER_LLM,
                tier=Tier.HEAVY,
                worker_module="mod2",
                modal_app_var="app2",
            ),
        ]

        targets = _group_deploy_targets(models)
        assert len(targets) == 2

        # mod1, app1 -> [m1, m2]
        # mod2, app2 -> [m3]
        target_map = {(t[0], t[1]): t[2] for t in targets}
        assert target_map[("mod1", "app1")] == ["m1", "m2"]
        assert target_map[("mod2", "app2")] == ["m3"]

    def test_group_deploy_targets_skips_missing_configs(self) -> None:
        """Should skip models without worker_module or modal_app_var."""
        from ai_workers.cli.deploy import _group_deploy_targets
        from ai_workers.common.config import ModelConfig, Task, Tier

        models = [
            ModelConfig(
                name="m1",
                hf_id="h1",
                task=Task.EMBEDDING,
                tier=Tier.LIGHT,
                worker_module="",
                modal_app_var="app1",
            ),
            ModelConfig(
                name="m2",
                hf_id="h2",
                task=Task.EMBEDDING,
                tier=Tier.HEAVY,
                worker_module="mod1",
                modal_app_var="",
            ),
        ]

        targets = _group_deploy_targets(models)
        assert len(targets) == 0


class TestDeploySingleSkip:
    """Test skipping deployment when config attributes are missing."""

    @patch("ai_workers.cli.deploy._deploy_app")
    @patch("ai_workers.cli.deploy.get_model")
    def test_skip_missing_worker_module(
        self, mock_get_model: MagicMock, mock_deploy_app: MagicMock
    ) -> None:
        """Should skip if worker_module is missing."""
        mock_config = MagicMock()
        mock_config.name = "dummy-model"
        mock_config.worker_module = ""
        mock_config.modal_app_var = "dummy_app"
        mock_get_model.return_value = mock_config

        _deploy_single("dummy-model")
        mock_deploy_app.assert_not_called()

    @patch("ai_workers.cli.deploy._deploy_app")
    @patch("ai_workers.cli.deploy.get_model")
    def test_skip_missing_modal_app_var(
        self, mock_get_model: MagicMock, mock_deploy_app: MagicMock
    ) -> None:
        """Should skip if modal_app_var is missing."""
        mock_config = MagicMock()
        mock_config.name = "dummy-model"
        mock_config.worker_module = "ai_workers.workers.dummy"
        mock_config.modal_app_var = ""
        mock_get_model.return_value = mock_config

        _deploy_single("dummy-model")
        mock_deploy_app.assert_not_called()
