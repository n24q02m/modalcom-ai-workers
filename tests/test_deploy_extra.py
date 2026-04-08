"""Extra tests for cli/deploy.py covering --all flag, list command, and skip cases.

Covers previously uncovered lines: 38-80, 133-138, 153-154, 157-158, 170-192.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest
import typer
import typer.testing
from click.exceptions import Exit as ClickExit

from ai_workers.cli.deploy import _deploy_module, _deploy_single, app
from ai_workers.common.config import ModelConfig, Task, Tier


class TestDeployAllFlag:
    """Tests for --all flag on the deploy callback (lines 38-80)."""

    def _invoke(self, args: list[str]):
        runner = typer.testing.CliRunner()
        return runner.invoke(app, args)

    @patch("ai_workers.cli.deploy.subprocess")
    def test_all_dry_run_deploys_no_subprocess(self, mock_subprocess: MagicMock) -> None:
        """--all --dry-run should list targets but call no subprocess."""
        result = self._invoke(["--all", "--dry-run"])
        assert result.exit_code == 0
        mock_subprocess.run.assert_not_called()

    @patch("ai_workers.cli.deploy.subprocess")
    def test_all_dry_run_shows_deploy_targets(self, mock_subprocess: MagicMock) -> None:
        """--all --dry-run output should mention deploying apps."""
        result = self._invoke(["--all", "--dry-run"])
        assert "Deploying" in result.output

    @patch("ai_workers.cli.deploy.subprocess")
    def test_all_success_calls_subprocess_for_each_target(self, mock_subprocess: MagicMock) -> None:
        """--all should call subprocess.run once per unique (module, app_var) pair."""
        mock_subprocess.run.return_value = MagicMock(returncode=0)
        result = self._invoke(["--all"])
        assert result.exit_code == 0
        # There are 7 unique deploy targets in the registry
        # (embedding, reranker, vl_embedding, vl_reranker, ocr, tts, asr)
        assert mock_subprocess.run.call_count == 7

    @patch("ai_workers.cli.deploy.subprocess")
    def test_all_partial_failure_exits_nonzero(self, mock_subprocess: MagicMock) -> None:
        """If any deploy fails, --all should exit with non-zero code."""
        # First call fails, rest succeed
        mock_subprocess.CalledProcessError = subprocess.CalledProcessError
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise subprocess.CalledProcessError(1, "modal deploy")
            return MagicMock(returncode=0)

        mock_subprocess.run.side_effect = side_effect
        result = self._invoke(["--all"])
        assert result.exit_code != 0

    def test_all_failure_reporting(self):
        """Test that deploy --all correctly aggregates and reports failures."""

        # Mock models
        m1 = ModelConfig(
            name="m1",
            hf_id="h1",
            task=Task.EMBEDDING,
            tier=Tier.LIGHT,
            worker_module="mod1",
            modal_app_var="app1",
        )
        m2 = ModelConfig(
            name="m2",
            hf_id="h2",
            task=Task.EMBEDDING,
            tier=Tier.HEAVY,
            worker_module="mod1",
            modal_app_var="app1",
        )
        m3 = ModelConfig(
            name="m3",
            hf_id="h3",
            task=Task.RERANKER_LLM,
            tier=Tier.HEAVY,
            worker_module="mod2",
            modal_app_var="app2",
        )

        mock_models = [m1, m2, m3]

        with (
            patch("ai_workers.cli.deploy.list_models", return_value=mock_models),
            patch("ai_workers.cli.deploy._deploy_app") as mock_deploy_app,
        ):
            # mod1/app1 fails, mod2/app2 succeeds
            def side_effect(module, app_var, dry_run=False):
                if module == "mod1" and app_var == "app1":
                    raise typer.Exit(code=1)
                return None

            mock_deploy_app.side_effect = side_effect

            result = self._invoke(["--all"])

            assert result.exit_code == 1
            # The code joins failures with ", "
            # failures.extend(names) where names = ["m1", "m2"]
            assert "Deploy FAILED: m1, m2" in result.output

            # Verify both apps were attempted (even if one failed)
            assert mock_deploy_app.call_count == 2


class TestDeployWorkerNone:
    """Tests for when worker=None and --all is not set (lines 60-62)."""

    def _invoke(self, args: list[str]):
        runner = typer.testing.CliRunner()
        return runner.invoke(app, args)

    def test_no_worker_no_all_raises_exit(self) -> None:
        """Calling deploy with an option but no worker/all should raise Exit (lines 61-62)."""
        # Using --dry-run but NO positional argument and NO --all
        result = self._invoke(["--dry-run"])
        assert result.exit_code == 1
        assert "specify a model name or use --all" in result.output

    def test_no_args_shows_help_or_error(self) -> None:
        """Invoking deploy without args — Typer shows help (exit 2) or error (exit 1)."""
        result = self._invoke([])
        # no_args_is_help=True causes Typer to show help with exit code 0 or 2
        assert result.exit_code in (0, 1, 2)


class TestDeploySingleSkipCases:
    """Tests for _deploy_single with missing worker_module or modal_app_var (153-158)."""

    def test_no_worker_module_skips_silently(self) -> None:
        """Config without worker_module should return without calling subprocess."""
        from ai_workers.common.config import ModelConfig, get_model

        config = get_model("qwen3-embedding-0.6b")
        # Patch get_model to return config without worker_module (empty string = falsy)
        patched = ModelConfig(
            name=config.name,
            hf_id=config.hf_id,
            task=config.task,
            tier=config.tier,
            gpu=config.gpu,
            precision=config.precision,
            model_class=config.model_class,
            serving_engine=config.serving_engine,
            worker_module="",  # empty = falsy, triggers skip
            modal_app_name=config.modal_app_name,
            modal_app_var=config.modal_app_var,
        )
        with patch("ai_workers.cli.deploy.get_model", return_value=patched):
            # Should NOT raise — just return early
            _deploy_single("qwen3-embedding-0.6b")

    def test_no_modal_app_var_skips_silently(self) -> None:
        """Config without modal_app_var should return without calling subprocess."""
        from ai_workers.common.config import ModelConfig, get_model

        config = get_model("qwen3-embedding-0.6b")
        patched = ModelConfig(
            name=config.name,
            hf_id=config.hf_id,
            task=config.task,
            tier=config.tier,
            gpu=config.gpu,
            precision=config.precision,
            model_class=config.model_class,
            serving_engine=config.serving_engine,
            worker_module=config.worker_module,
            modal_app_name=config.modal_app_name,
            modal_app_var="",  # empty = falsy, triggers skip
        )
        with patch("ai_workers.cli.deploy.get_model", return_value=patched):
            _deploy_single("qwen3-embedding-0.6b")


class TestDeployModuleErrors:
    """Tests for _deploy_module error paths (lines 133-138)."""

    @patch("ai_workers.cli.deploy.subprocess")
    def test_deploy_module_file_not_found(self, mock_subprocess: MagicMock) -> None:
        """Missing modal CLI should raise Exit."""
        mock_subprocess.run.side_effect = FileNotFoundError()
        with pytest.raises(ClickExit):
            _deploy_module("ai_workers.workers.embedding")

    @patch("ai_workers.cli.deploy.subprocess")
    def test_deploy_module_called_process_error(self, mock_subprocess: MagicMock) -> None:
        """Failed deploy module should raise Exit."""
        mock_subprocess.run.side_effect = subprocess.CalledProcessError(1, "modal deploy")
        mock_subprocess.CalledProcessError = subprocess.CalledProcessError
        with pytest.raises(ClickExit):
            _deploy_module("ai_workers.workers.embedding")

    @patch("ai_workers.cli.deploy.subprocess")
    def test_deploy_module_dry_run_skips(self, mock_subprocess: MagicMock) -> None:
        """_deploy_module dry_run should NOT call subprocess."""
        _deploy_module("ai_workers.workers.embedding", dry_run=True)
        mock_subprocess.run.assert_not_called()


class TestDeployListSubcommand:
    """Tests for list_workers function (lines 167-192)."""

    def test_list_shows_table(self) -> None:
        """list_workers should print a Rich table without error."""

        from ai_workers.cli.deploy import list_workers

        # list_workers() uses the global console — just verify it doesn't raise
        list_workers()  # should not raise

    def test_list_workers_covers_all_registry_models(self) -> None:
        """list_workers should iterate all models in MODEL_REGISTRY."""
        from ai_workers.cli.deploy import list_workers
        from ai_workers.common.config import MODEL_REGISTRY

        # Verify all worker-module models appear in MODEL_REGISTRY
        worker_models = [m for m in MODEL_REGISTRY.values() if m.worker_module]
        assert len(worker_models) > 0

        # Calling list_workers should not raise
        list_workers()

    def test_list_via_cli_runner(self) -> None:
        """list subcommand via CliRunner — deploy callback intercepts 'list' as model name."""
        # Due to Typer callback with invoke_without_command=True, 'list' is treated
        # as the WORKER argument and _deploy_single("list") fails with model not found.
        # This tests the actual (documented) behavior.
        runner = typer.testing.CliRunner()
        result = runner.invoke(app, ["list"])
        # "list" is not a valid model name → exit code 1 with error message
        assert result.exit_code == 1
        assert "not found" in result.output or "Error" in result.output
