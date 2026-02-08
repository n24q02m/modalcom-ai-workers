"""Convert HuggingFace models to SafeTensors at target precision.

Runs locally on CPU (+ optional GPU offload for large models).
Downloads from HuggingFace Hub -> casts to FP16/BF16 -> saves SafeTensors.

Hardware requirements (4GB VRAM + 16GB RAM):
  - 0.6B models: ~1.2 GB RAM (CPU only)
  - 2B models:   ~4 GB RAM (CPU only)
  - 3B models:   ~6 GB RAM (CPU only)
  - 8B models:   ~16 GB RAM (CPU + GPU offload recommended)
"""

from __future__ import annotations

import gc
import warnings
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ai_workers.common.config import (
    MODEL_REGISTRY,
    Task,
    get_model,
    get_model_class,
    get_torch_dtype,
    list_models,
)

app = typer.Typer(no_args_is_help=True)
console = Console()

# Default output directory (relative to project root)
DEFAULT_OUTPUT_DIR = Path("converted")


def _clear_memory() -> None:
    """Free RAM and GPU memory."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _is_audio_model(task: Task) -> bool:
    return task == Task.ASR


def _is_vl_model(task: Task) -> bool:
    return task in {Task.VL_EMBEDDING, Task.VL_RERANKER}


@app.callback(invoke_without_command=True)
def convert(
    model: str = typer.Argument(
        ...,
        help="Model registry name (e.g. 'qwen3-embedding-0.6b') or 'all'",
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR,
        "--output-dir",
        "-o",
        help="Output directory for converted models",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing converted model",
    ),
) -> None:
    """Convert a HuggingFace model to SafeTensors at target precision."""
    warnings.filterwarnings("ignore")

    if model == "all":
        models = list_models()
        console.print(f"[bold]Converting all {len(models)} models...[/bold]")
        for m in models:
            _convert_single(m.name, output_dir, force=force)
        return

    _convert_single(model, output_dir, force=force)


def _convert_single(model_name: str, output_dir: Path, *, force: bool = False) -> None:
    """Convert a single model."""
    try:
        config = get_model(model_name)
    except KeyError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1) from None

    model_output = output_dir / config.name
    if model_output.exists() and not force:
        console.print(
            f"[yellow]Skipping {config.name} — already exists at {model_output}. "
            f"Use --force to overwrite.[/yellow]"
        )
        return

    console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
    console.print(f"[bold]Converting: {config.name}[/bold]")
    console.print(f"  HuggingFace ID: {config.hf_id}")
    console.print(f"  Precision: {config.precision.value}")
    console.print(f"  Model Class: {config.model_class.value}")
    console.print(f"  Output: {model_output}")
    console.print(f"[bold cyan]{'=' * 60}[/bold cyan]")

    model_output.mkdir(parents=True, exist_ok=True)

    try:
        # Determine torch dtype
        dtype = get_torch_dtype(config.precision)
        console.print(f"\n[dim]Loading model with dtype={dtype}...[/dim]")

        # Build load kwargs
        load_kwargs: dict[str, object] = {
            "trust_remote_code": config.trust_remote_code,
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
            "device_map": "cpu",  # CPU-only conversion for safety
            **config.extra_load_kwargs,
        }

        # Load model
        model_cls = get_model_class(config.model_class)
        model = model_cls.from_pretrained(config.hf_id, **load_kwargs)

        # Load tokenizer/processor
        console.print("[dim]Loading tokenizer/processor...[/dim]")
        if _is_audio_model(config.task) or _is_vl_model(config.task):
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(
                config.hf_id,
                trust_remote_code=config.trust_remote_code,
            )
        else:
            from transformers import AutoTokenizer

            processor = AutoTokenizer.from_pretrained(
                config.hf_id,
                trust_remote_code=config.trust_remote_code,
            )

        # Save as SafeTensors
        console.print(f"[dim]Saving SafeTensors to {model_output}...[/dim]")
        model.save_pretrained(model_output, safe_serialization=True)
        processor.save_pretrained(model_output)

        # Report file sizes
        total_size = 0.0
        table = Table(title=f"Output files: {config.name}")
        table.add_column("File", style="cyan")
        table.add_column("Size", justify="right", style="green")

        for f in sorted(model_output.iterdir()):
            if f.is_file():
                size_mb = f.stat().st_size / (1024**2)
                total_size += size_mb
                table.add_row(f.name, f"{size_mb:.2f} MB")

        table.add_row("[bold]Total[/bold]", f"[bold]{total_size:.2f} MB[/bold]")
        console.print(table)
        console.print(f"[green]Convert {config.name}: SUCCESS[/green]\n")

        # Cleanup
        del model
        _clear_memory()

    except Exception as e:
        console.print(f"[red]Convert {config.name}: FAILED — {e}[/red]")
        import traceback

        traceback.print_exc()
        _clear_memory()
        raise typer.Exit(code=1) from None


@app.command("list")
def list_available() -> None:
    """List all available models in the registry."""
    table = Table(title="Model Registry")
    table.add_column("Name", style="cyan")
    table.add_column("HuggingFace ID", style="dim")
    table.add_column("Task")
    table.add_column("Tier")
    table.add_column("Precision")
    table.add_column("GPU")

    for name in sorted(MODEL_REGISTRY):
        m = MODEL_REGISTRY[name]
        table.add_row(
            m.name,
            m.hf_id,
            m.task.value,
            m.tier.value,
            m.precision.value,
            m.gpu.value,
        )

    console.print(table)
