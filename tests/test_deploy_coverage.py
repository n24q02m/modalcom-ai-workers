from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
from click.exceptions import Exit as ClickExit
from ai_workers.cli.deploy import deploy, list_workers
from ai_workers.common.config import ModelConfig, Task, Tier, GPU, ServingEngine

def test_deploy_no_worker_error() -> None:
    """Calling deploy without worker or --all should raise Exit."""
    with patch("ai_workers.cli.deploy.console") as mock_console:
        with pytest.raises(ClickExit) as excinfo:
            deploy(worker=None, all_workers=False, dry_run=False)
        assert excinfo.value.exit_code == 1
        mock_console.print.assert_called_with("[red]Error: specify a model name or use --all[/red]")

@patch("ai_workers.cli.deploy.list_models")
@patch("ai_workers.cli.deploy._deploy_app")
@patch("ai_workers.cli.deploy.console")
def test_deploy_all_success(mock_console: MagicMock, mock_deploy_app: MagicMock, mock_list_models: MagicMock) -> None:
    """deploy --all should call _deploy_app for each unique module/app_var."""
    mock_list_models.return_value = [
        ModelConfig(
            name="m1", hf_id="h1", task=Task.EMBEDDING, tier=Tier.LIGHT,
            worker_module="mod1", modal_app_var="app1",
            modal_app_name="n1", gpu=GPU.T4, serving_engine=ServingEngine.CUSTOM_FASTAPI
        ),
        ModelConfig(
            name="m2", hf_id="h2", task=Task.EMBEDDING, tier=Tier.HEAVY,
            worker_module="mod1", modal_app_var="app1",
            modal_app_name="n1", gpu=GPU.T4, serving_engine=ServingEngine.CUSTOM_FASTAPI
        ),
        ModelConfig(
            name="m3", hf_id="h3", task=Task.RERANKER_LLM, tier=Tier.HEAVY,
            worker_module="mod2", modal_app_var="app2",
            modal_app_name="n2", gpu=GPU.A10G, serving_engine=ServingEngine.VLLM
        ),
    ]
    deploy(worker=None, all_workers=True, dry_run=True)
    assert mock_deploy_app.call_count == 2
    mock_deploy_app.assert_any_call("mod1", "app1", dry_run=True)
    mock_deploy_app.assert_any_call("mod2", "app2", dry_run=True)

@patch("ai_workers.cli.deploy.list_models")
@patch("ai_workers.cli.deploy._deploy_app")
@patch("ai_workers.cli.deploy.console")
def test_deploy_all_failure(mock_console: MagicMock, mock_deploy_app: MagicMock, mock_list_models: MagicMock) -> None:
    """deploy --all should report failures and exit with 1."""
    mock_list_models.return_value = [
        ModelConfig(
            name="m1", hf_id="h1", task=Task.EMBEDDING, tier=Tier.LIGHT,
            worker_module="mod1", modal_app_var="app1",
            modal_app_name="n1", gpu=GPU.T4, serving_engine=ServingEngine.CUSTOM_FASTAPI
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
def test_list_workers(mock_console: MagicMock, mock_registry: MagicMock) -> None:
    """list_workers should print a table of models."""
    mock_registry.__iter__.return_value = ["m1"]
    mock_registry.__getitem__.return_value = ModelConfig(
        name="m1", hf_id="h1", task=Task.EMBEDDING, tier=Tier.LIGHT,
        worker_module="mod1", modal_app_var="app1",
        modal_app_name="n1", gpu=GPU.T4, serving_engine=ServingEngine.CUSTOM_FASTAPI
    )
    list_workers()
    # verify that a table was printed
    from rich.table import Table
    table_called = False
    for call in mock_console.print.call_args_list:
        if isinstance(call.args[0], Table):
            table_called = True
            break
    assert table_called
