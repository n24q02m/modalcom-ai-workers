"""Upload converted models to Cloudflare R2 (and optionally GDrive backup).

Reads from ./converted/<model_name>/ and uploads to R2 bucket.
"""

from __future__ import annotations

from pathlib import Path

import typer  # type: ignore
from rich.console import Console

from ai_workers.common.config import get_model, list_models
from ai_workers.common.r2 import R2Config, upload_directory

app = typer.Typer(no_args_is_help=True)
console = Console()

DEFAULT_CONVERTED_DIR = Path("converted")


@app.callback(invoke_without_command=True)
def upload(
    model: str = typer.Argument(
        ...,
        help="Model registry name (e.g. 'qwen3-embedding-0.6b') or 'all'",
    ),
    converted_dir: Path = typer.Option(
        DEFAULT_CONVERTED_DIR,
        "--converted-dir",
        "-d",
        help="Directory containing converted models",
    ),
    backup_gdrive: bool = typer.Option(
        False,
        "--backup-gdrive",
        help="Also sync to GDrive via rclone (requires rclone configured)",
    ),
) -> None:
    """Upload a converted model to Cloudflare R2."""
    if model == "all":
        models = list_models()
        console.print(f"[bold]Uploading all {len(models)} models...[/bold]")
        for m in models:
            _upload_single(m.name, converted_dir, backup_gdrive=backup_gdrive)
        return

    _upload_single(model, converted_dir, backup_gdrive=backup_gdrive)


def _upload_single(
    model_name: str,
    converted_dir: Path,
    *,
    backup_gdrive: bool = False,
) -> None:
    """Upload a single model to R2."""
    try:
        config = get_model(model_name)
    except KeyError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1) from None

    local_dir = converted_dir / config.name
    if not local_dir.exists():
        console.print(f"[red]Error: {local_dir} does not exist. Run 'convert' first.[/red]")
        raise typer.Exit(code=1) from None

    console.print(f"\n[bold cyan]Uploading {config.name} to R2...[/bold cyan]")
    console.print(f"  Source: {local_dir}")
    console.print(f"  R2 prefix: {config.r2_prefix}/")

    try:
        r2_config = R2Config.from_env()
        count = upload_directory(local_dir, config.r2_prefix, r2_config)
        console.print(f"[green]Upload {config.name}: SUCCESS ({count} files)[/green]")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1) from None
    except Exception as e:
        console.print(f"[red]Upload {config.name}: FAILED — {e}[/red]")
        raise typer.Exit(code=1) from None

    # Optional GDrive backup via rclone
    if backup_gdrive:
        _sync_gdrive(local_dir, config.r2_prefix)


def _sync_gdrive(local_dir: Path, prefix: str) -> None:
    """Sync model to GDrive using rclone."""
    import subprocess

    gdrive_remote = f"gdrive:ai-workers-models/{prefix}"
    console.print(f"[dim]Syncing to GDrive: {gdrive_remote}[/dim]")

    try:
        result = subprocess.run(
            ["rclone", "sync", str(local_dir), gdrive_remote, "--progress"],
            check=True,
            capture_output=False,
        )
        if result.returncode == 0:
            console.print("[green]GDrive backup: SUCCESS[/green]")
    except FileNotFoundError:
        console.print("[yellow]rclone not found. Install rclone for GDrive backup.[/yellow]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]GDrive backup: FAILED — {e}[/red]")


@app.command("list")
def list_available() -> None:
    """List models available for upload (already converted)."""
    converted_dir = DEFAULT_CONVERTED_DIR
    if not converted_dir.exists():
        console.print("[yellow]No converted models found. Run 'convert' first.[/yellow]")
        return

    from rich.table import Table

    table = Table(title="Converted Models (ready for upload)")
    table.add_column("Name", style="cyan")
    table.add_column("Size", justify="right", style="green")
    table.add_column("Files", justify="right")

    for model_dir in sorted(converted_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        files = list(model_dir.rglob("*"))
        file_count = sum(1 for f in files if f.is_file())
        total_size = sum(f.stat().st_size for f in files if f.is_file()) / (1024**2)
        table.add_row(model_dir.name, f"{total_size:.1f} MB", str(file_count))

    console.print(table)
