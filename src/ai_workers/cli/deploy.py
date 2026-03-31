"""Deploy Modal workers via CLI.

Wraps `modal deploy` for individual or all workers.
Supports multi-app worker files by specifying the app variable name.
"""

from __future__ import annotations

import subprocess

import typer
from rich.console import Console

from ai_workers.common.config import MODEL_REGISTRY, ModelConfig, get_model, list_models

app = typer.Typer(no_args_is_help=True)
console = Console()


def _group_deploy_targets(models: list[ModelConfig]) -> list[tuple[str, str, list[str]]]:
    """Group models by (worker_module, modal_app_var)."""
    seen: set[tuple[str, str]] = set()
    deploy_targets: list[tuple[str, str, list[str]]] = []  # (module, app_var, model_names)

    for m in models:
        if not m.worker_module or not m.modal_app_var:
            continue
        key = (m.worker_module, m.modal_app_var)
        if key not in seen:
            seen.add(key)
            deploy_targets.append((m.worker_module, m.modal_app_var, [m.name]))
        else:
            # Append model name to existing target
            for target in deploy_targets:
                if (target[0], target[1]) == key:
                    target[2].append(m.name)
                    break
    return deploy_targets


@app.callback(invoke_without_command=True)
def deploy(
    worker: str = typer.Argument(
        None,
        help="Model registry name to deploy (e.g. 'qwen3-embedding-0.6b') or omit for interactive",
    ),
    all_workers: bool = typer.Option(
        False,
        "--all",
        help="Deploy all workers",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show commands without executing",
    ),
) -> None:
    """Deploy a worker to Modal.com."""
    if all_workers:
        # Group by (worker_module, modal_app_var) to deploy each unique app once
        models = list_models()
        deploy_targets = _group_deploy_targets(models)

        console.print(
            f"[bold]Deploying {len(deploy_targets)} apps ({len(models)} models)...[/bold]"
        )
        for module, app_var, names in deploy_targets:
            console.print(f"  {app_var} ({module}) -> {', '.join(names)}")

        failures: list[str] = []
        for module, app_var, names in deploy_targets:
            try:
                _deploy_app(module, app_var, dry_run=dry_run)
            except typer.Exit:
                failures.extend(names)

        if failures:
            console.print(f"\n[red bold]Deploy FAILED: {', '.join(failures)}[/red bold]")
            raise typer.Exit(code=1)
        return

    if worker is None:
        console.print("[red]Error: specify a model name or use --all[/red]")
        raise typer.Exit(code=1) from None

    _deploy_single(worker, dry_run=dry_run)


def _module_to_file_path(module: str) -> str:
    """Convert a Python module path to a file path relative to project root."""
    return "src/" + module.replace(".", "/") + ".py"


def _deploy_app(module: str, app_var: str, *, dry_run: bool = False) -> None:
    """Deploy a specific app within a worker module file.

    Uses the ``modal deploy file.py::app_var`` format to specify
    the app when the file contains multiple apps.
    """
    file_path = _module_to_file_path(module)
    target = f"{file_path}::{app_var}"
    cmd = ["modal", "deploy", target]

    console.print(f"\n[bold cyan]Deploying: {app_var} ({module})[/bold cyan]")
    console.print(f"  Command: {' '.join(cmd)}")

    if dry_run:
        console.print("[yellow]  (dry run -- skipped)[/yellow]")
        return

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        if result.returncode == 0:
            console.print(f"[green]Deploy {app_var}: SUCCESS[/green]")
    except FileNotFoundError:
        console.print("[red]Error: `modal` CLI not found. Run `pip install modal`[/red]")
        raise typer.Exit(code=1) from None
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Deploy {app_var}: FAILED -- exit code {e.returncode}[/red]")
        raise typer.Exit(code=1) from None


def _deploy_module(module: str, *, dry_run: bool = False) -> None:
    """Deploy all apps in a worker module file."""
    file_path = _module_to_file_path(module)
    cmd = ["modal", "deploy", file_path]

    console.print(f"\n[bold cyan]Deploying module: {module}[/bold cyan]")
    console.print(f"  Command: {' '.join(cmd)}")

    if dry_run:
        console.print("[yellow]  (dry run -- skipped)[/yellow]")
        return

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        if result.returncode == 0:
            console.print(f"[green]Deploy {module}: SUCCESS[/green]")
    except FileNotFoundError:
        console.print("[red]Error: `modal` CLI not found. Run `pip install modal`[/red]")
        raise typer.Exit(code=1) from None
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Deploy {module}: FAILED -- exit code {e.returncode}[/red]")
        raise typer.Exit(code=1) from None


def _deploy_single(model_name: str, *, dry_run: bool = False) -> None:
    """Deploy a single worker by model name.

    Uses ``modal deploy file.py::app_var`` for the specific app.
    """
    try:
        config = get_model(model_name)
    except KeyError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1) from None

    if not config.worker_module:
        console.print(f"[yellow]Skipping {config.name} -- no worker_module configured[/yellow]")
        return

    if not config.modal_app_var:
        console.print(f"[yellow]Skipping {config.name} -- no modal_app_var configured[/yellow]")
        return

    console.print(f"\n[bold cyan]Deploying: {config.name}[/bold cyan]")
    console.print(f"  App: {config.modal_app_name}")
    console.print(f"  Module: {config.worker_module}")

    _deploy_app(config.worker_module, config.modal_app_var, dry_run=dry_run)


@app.command("list")
def list_workers() -> None:
    """List all deployable workers."""
    from rich.table import Table

    table = Table(title="Deployable Workers")
    table.add_column("Name", style="cyan")
    table.add_column("Modal App", style="dim")
    table.add_column("App Var")
    table.add_column("GPU")
    table.add_column("Engine")
    table.add_column("Module")

    for name in sorted(MODEL_REGISTRY):
        m = MODEL_REGISTRY[name]
        if m.worker_module:
            table.add_row(
                m.name,
                m.modal_app_name,
                m.modal_app_var,
                m.gpu.value,
                m.serving_engine.value,
                m.worker_module,
            )

    console.print(table)
