"""Convert HuggingFace models to ONNX multi-variant and push to HuggingFace Hub.

Runs on Modal CPU container (32GB RAM, 4 CPU cores).
Pipeline: HuggingFace Hub -> ONNX FP32 -> onnxslim -> INT8 + Q4F16 -> HF Hub (public).

Output fastembed-compatible:
  onnx/model_quantized.onnx (INT8) + onnx/model_q4f16.onnx (Q4F16) + tokenizer files.

Usage:
  python -m ai_workers onnx-convert list
  python -m ai_workers onnx-convert qwen3-embedding-0.6b-onnx
  python -m ai_workers onnx-convert all --force
"""

from __future__ import annotations

import modal
import typer
from rich.console import Console
from rich.table import Table

from ai_workers.workers.onnx_converter import ONNX_MODELS, onnx_convert_app, onnx_convert_model

app = typer.Typer(no_args_is_help=True)
console = Console(width=200)


@app.callback(invoke_without_command=True)
def onnx_convert(
    model: str = typer.Argument(
        None,
        help="ONNX model name (e.g. 'qwen3-embedding-0.6b-onnx'), 'all', or 'list'",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite if repo already exists on HF Hub",
    ),
) -> None:
    """Convert HuggingFace model to ONNX multi-variant (INT8 + Q4F16) on Modal CPU, push to HF Hub."""
    if model is None:
        console.print(
            "[yellow]Please specify a model, 'all', or 'list'. Use --help for usage guide.[/yellow]"
        )
        raise typer.Exit(code=1)

    if model == "list":
        list_onnx_models()
        return

    if model == "all":
        console.print(
            f"[bold]Converting {len(ONNX_MODELS)} models to ONNX (INT8 + Q4F16) on Modal CPU...[/bold]"
        )
        failed: list[str] = []
        for name in ONNX_MODELS:
            try:
                _onnx_convert_remote(name, force=force)
            except (SystemExit, Exception):
                failed.append(name)
        if failed:
            console.print(
                f"\n[red bold]{len(failed)} model(s) failed: {', '.join(failed)}[/red bold]"
            )
            raise typer.Exit(code=1)
        console.print(
            f"\n[green bold]All {len(ONNX_MODELS)} models converted successfully![/green bold]"
        )
        return

    _onnx_convert_remote(model, force=force)


def _onnx_convert_remote(model_name: str, *, force: bool = False) -> None:
    """Call Modal remote function to ONNX convert a model."""
    if model_name not in ONNX_MODELS:
        available = ", ".join(sorted(ONNX_MODELS.keys()))
        console.print(f"[red]Error: Model '{model_name}' not found. Available: {available}[/red]")
        raise typer.Exit(code=1)

    config = ONNX_MODELS[model_name]

    console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
    console.print(f"[bold]ONNX Converting: {config.name}[/bold]")
    console.print(f"  Source: {config.hf_source}")
    console.print(f"  Target: {config.hf_target}")
    console.print(f"  Model Class: {config.model_class}")
    console.print(f"  Output: {config.output_attr}")
    console.print("  Variants: INT8 (model_quantized.onnx) + Q4F16 (model_q4f16.onnx)")
    console.print("  Runs on: Modal CPU (32GB RAM)")
    console.print("  Output: HuggingFace Hub (public repo)")
    console.print(f"[bold cyan]{'=' * 60}[/bold cyan]")

    try:
        with modal.enable_output(), onnx_convert_app.run():
            result = onnx_convert_model.remote(
                model_name=config.name,
                hf_source=config.hf_source,
                hf_target=config.hf_target,
                model_class=config.model_class,
                output_attr=config.output_attr,
                trust_remote_code=config.trust_remote_code,
                force=force,
            )

        status = result.get("status", "unknown")
        if status == "skipped":
            console.print(
                f"[yellow]Skipped {config.name} — repo already exists on HF Hub. "
                f"Use --force to overwrite.[/yellow]"
            )
        elif status == "success":
            files_count = result.get("files_count", 0)
            total_size = result.get("total_size_mb", 0)
            url = result.get("url", "")
            variants = result.get("variants", {})
            console.print(
                f"[green]ONNX Convert {config.name}: SUCCESS "
                f"({files_count} files, {total_size:.2f} MB)[/green]"
            )
            for vname, vinfo in variants.items():
                if isinstance(vinfo, dict):
                    console.print(
                        f"  [dim]{vname}: {vinfo.get('file', '?')} "
                        f"({vinfo.get('size_mb', 0):.2f} MB)[/dim]"
                    )
            console.print(f"  [dim]{url}[/dim]")
        else:
            console.print(f"[red]ONNX Convert {config.name}: unknown status — {result}[/red]")
            raise typer.Exit(code=1) from None

    except modal.exception.AuthError:
        console.print("[red]Error: Modal not authenticated. Run `modal token set` first.[/red]")
        raise typer.Exit(code=1) from None
    except Exception as e:
        console.print(f"[red]ONNX Convert {config.name}: FAILED — {e}[/red]")
        import traceback

        traceback.print_exc()
        raise typer.Exit(code=1) from None


@app.command("list")
def list_onnx_models() -> None:
    """List all convertible ONNX models."""
    table = Table(title="ONNX Model Registry")
    table.add_column("Name", style="cyan")
    table.add_column("Source (HF)", style="dim")
    table.add_column("Target (HF)")
    table.add_column("Model Class")
    table.add_column("Output")

    for name in sorted(ONNX_MODELS):
        config = ONNX_MODELS[name]
        table.add_row(
            config.name,
            config.hf_source,
            config.hf_target,
            config.model_class,
            config.output_attr,
        )

    console.print(table)
