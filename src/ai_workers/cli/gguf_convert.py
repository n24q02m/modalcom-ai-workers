"""Convert HuggingFace models to GGUF Q4_K_M and push to HuggingFace Hub.

Runs on Modal CPU container (32GB RAM, 4 CPU cores).
Pipeline: HuggingFace Hub -> convert_hf_to_gguf.py (F16) -> llama-quantize (Q4_K_M) -> HF Hub.

Output: gguf/{model_name}-q4-k-m.gguf

Usage:
  python -m ai_workers gguf-convert list
  python -m ai_workers gguf-convert qwen3-embedding-0.6b-gguf
  python -m ai_workers gguf-convert all --force
"""


import modal
import typer
from rich.console import Console
from rich.table import Table

from ai_workers.workers.gguf_converter import GGUF_MODELS, gguf_convert_app, gguf_convert_model

app = typer.Typer(no_args_is_help=True)
console = Console(width=200)


@app.callback(invoke_without_command=True)
def gguf_convert(
    model: str = typer.Argument(
        None,
        help="GGUF model name (e.g. 'qwen3-embedding-0.6b-gguf'), 'all', or 'list'",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite if file already exists on HF Hub",
    ),
    quant_type: str = typer.Option(
        "Q4_K_M",
        "--quant",
        "-q",
        help="GGUF quantization type (default: Q4_K_M)",
    ),
) -> None:
    """Convert HuggingFace model to GGUF on Modal CPU, push to HF Hub."""
    if model is None:
        console.print(
            "[yellow]Please specify a model, 'all', or 'list'. Use --help for usage guide.[/yellow]"
        )
        raise typer.Exit(code=1)

    if model == "list":
        list_gguf_models()
        return

    if model == "all":
        console.print(
            f"[bold]Converting {len(GGUF_MODELS)} models to GGUF {quant_type} on Modal CPU...[/bold]"
        )
        failed: list[str] = []
        for name in GGUF_MODELS:
            try:
                _gguf_convert_remote(name, force=force, quant_type=quant_type)
            except (SystemExit, Exception):
                failed.append(name)
        if failed:
            console.print(
                f"\n[red bold]{len(failed)} model(s) failed: {', '.join(failed)}[/red bold]"
            )
            raise typer.Exit(code=1)
        console.print(
            f"\n[green bold]All {len(GGUF_MODELS)} models converted successfully![/green bold]"
        )
        return

    _gguf_convert_remote(model, force=force, quant_type=quant_type)


def _gguf_convert_remote(
    model_name: str, *, force: bool = False, quant_type: str = "Q4_K_M"
) -> None:
    """Call Modal remote function to GGUF convert a model."""
    if model_name not in GGUF_MODELS:
        available = ", ".join(sorted(GGUF_MODELS.keys()))
        console.print(f"[red]Error: Model '{model_name}' not found. Available: {available}[/red]")
        raise typer.Exit(code=1)

    config = GGUF_MODELS[model_name]

    console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
    console.print(f"[bold]GGUF Converting: {config.name}[/bold]")
    console.print(f"  Source: {config.hf_source}")
    console.print(f"  Target: {config.hf_target}")
    console.print(f"  GGUF Name: {config.gguf_name}")
    console.print(f"  Quantization: {quant_type}")
    console.print("  Runs on: Modal CPU (32GB RAM)")
    console.print("  Output: HuggingFace Hub (existing repo)")
    console.print(f"[bold cyan]{'=' * 60}[/bold cyan]")

    try:
        with modal.enable_output(), gguf_convert_app.run():
            result = gguf_convert_model.remote(
                model_name=config.name,
                hf_source=config.hf_source,
                hf_target=config.hf_target,
                gguf_name=config.gguf_name,
                output_attr=config.output_attr,
                quant_type=quant_type,
                force=force,
            )

        status = result.get("status", "unknown")
        if status == "skipped":
            console.print(
                f"[yellow]Skipped {config.name} — file already exists on HF Hub. "
                f"Use --force to overwrite.[/yellow]"
            )
        elif status == "success":
            size_mb = result.get("size_mb", 0)
            gguf_file = result.get("gguf_file", "")
            url = result.get("url", "")
            console.print(
                f"[green]GGUF Convert {config.name}: SUCCESS "
                f"({gguf_file}, {size_mb:.2f} MB)[/green]"
            )
            console.print(f"  [dim]{url}[/dim]")
        else:
            console.print(f"[red]GGUF Convert {config.name}: unknown status — {result}[/red]")
            raise typer.Exit(code=1) from None

    except modal.exception.AuthError:
        console.print("[red]Error: Modal not authenticated. Run `modal token set` first.[/red]")
        raise typer.Exit(code=1) from None
    except Exception as e:
        console.print(f"[red]GGUF Convert {config.name}: FAILED — {e}[/red]")
        import traceback

        traceback.print_exc()
        raise typer.Exit(code=1) from None


@app.command("list")
def list_gguf_models() -> None:
    """List all convertible GGUF models."""
    table = Table(title="GGUF Model Registry")
    table.add_column("Name", style="cyan")
    table.add_column("Source (HF)", style="dim")
    table.add_column("Target (HF)")
    table.add_column("GGUF Name")

    for name in sorted(GGUF_MODELS):
        config = GGUF_MODELS[name]
        table.add_row(
            config.name,
            config.hf_source,
            config.hf_target,
            config.gguf_name,
        )

    console.print(table)
