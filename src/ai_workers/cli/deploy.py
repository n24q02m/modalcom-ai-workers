"""Deploy Modal workers via CLI.

Wraps `modal deploy` for individual or all workers.
"""

from __future__ import annotations

import subprocess

import typer
from rich.console import Console

from ai_workers.common.config import MODEL_REGISTRY, get_model, list_models

app = typer.Typer(no_args_is_help=True)
console = Console()


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
        models = list_models()
        console.print(f"[bold]Deploying all {len(models)} workers...[/bold]")
        for m in models:
            _deploy_single(m.name, dry_run=dry_run)
        return

    if worker is None:
        console.print("[red]Error: specify a model name or use --all[/red]")
        raise typer.Exit(code=1) from None

    _deploy_single(worker, dry_run=dry_run)


def _deploy_single(model_name: str, *, dry_run: bool = False) -> None:
    """Deploy a single worker."""
    try:
        config = get_model(model_name)
    except KeyError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1) from None

    if not config.worker_module:
        console.print(f"[yellow]Skipping {config.name} — no worker_module configured[/yellow]")
        return

    # Convert Python module path to file path
    module_path = config.worker_module.replace(".", "/") + ".py"
    # The apps are defined as module-level variables; modal deploy uses the module path
    cmd = ["modal", "deploy", f"src/{module_path}"]

    console.print(f"\n[bold cyan]Deploying: {config.name}[/bold cyan]")
    console.print(f"  Command: {' '.join(cmd)}")

    if dry_run:
        console.print("[yellow]  (dry run — skipped)[/yellow]")
        return

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        if result.returncode == 0:
            console.print(f"[green]Deploy {config.name}: SUCCESS[/green]")
    except FileNotFoundError:
        console.print("[red]Error: `modal` CLI not found. Run `pip install modal`[/red]")
        raise typer.Exit(code=1) from None
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Deploy {config.name}: FAILED — exit code {e.returncode}[/red]")
        raise typer.Exit(code=1) from None


@app.command("list")
def list_workers() -> None:
    """List all deployable workers."""
    from rich.table import Table

    table = Table(title="Deployable Workers")
    table.add_column("Name", style="cyan")
    table.add_column("Modal App", style="dim")
    table.add_column("GPU")
    table.add_column("Engine")
    table.add_column("Module")

    for name in sorted(MODEL_REGISTRY):
        m = MODEL_REGISTRY[name]
        if m.worker_module:
            table.add_row(
                m.name,
                m.modal_app_name,
                m.gpu.value,
                m.serving_engine.value,
                m.worker_module,
            )

    console.print(table)
