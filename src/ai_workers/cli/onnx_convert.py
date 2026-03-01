"""Convert HuggingFace models sang ONNX multi-variant va push len HuggingFace Hub.

Chay tren Modal CPU container (32GB RAM, 4 CPU cores).
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
        help="Tên model ONNX (vd: 'qwen3-embedding-0.6b-onnx'), 'all', hoặc 'list'",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Ghi đè nếu repo đã tồn tại trên HF Hub",
    ),
) -> None:
    """Convert model HuggingFace sang ONNX multi-variant (INT8 + Q4F16) tren Modal CPU, push len HF Hub."""
    if model is None:
        console.print(
            "[yellow]Cần chỉ định model, 'all', hoặc 'list'. Dùng --help để xem hướng dẫn.[/yellow]"
        )
        raise typer.Exit(code=1)

    if model == "list":
        list_onnx_models()
        return

    if model == "all":
        console.print(
            f"[bold]Converting {len(ONNX_MODELS)} models sang ONNX (INT8 + Q4F16) tren Modal CPU...[/bold]"
        )
        failed: list[str] = []
        for name in ONNX_MODELS:
            try:
                _onnx_convert_remote(name, force=force)
            except (SystemExit, Exception):
                failed.append(name)
        if failed:
            console.print(
                f"\n[red bold]{len(failed)} model(s) thất bại: {', '.join(failed)}[/red bold]"
            )
            raise typer.Exit(code=1)
        console.print(
            f"\n[green bold]Tất cả {len(ONNX_MODELS)} models convert thành công![/green bold]"
        )
        return

    _onnx_convert_remote(model, force=force)


def _onnx_convert_remote(model_name: str, *, force: bool = False) -> None:
    """Gọi Modal remote function để ONNX convert một model."""
    if model_name not in ONNX_MODELS:
        available = ", ".join(sorted(ONNX_MODELS.keys()))
        console.print(
            f"[red]Lỗi: Model '{model_name}' không tìm thấy. Available: {available}[/red]"
        )
        raise typer.Exit(code=1)

    config = ONNX_MODELS[model_name]

    console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
    console.print(f"[bold]ONNX Converting: {config.name}[/bold]")
    console.print(f"  Source: {config.hf_source}")
    console.print(f"  Target: {config.hf_target}")
    console.print(f"  Model Class: {config.model_class}")
    console.print(f"  Output: {config.output_attr}")
    console.print("  Variants: INT8 (model_quantized.onnx) + Q4F16 (model_q4f16.onnx)")
    console.print("  Chay tren: Modal CPU (32GB RAM)")
    console.print("  Dau ra: HuggingFace Hub (public repo)")
    console.print(f"[bold cyan]{'=' * 60}[/bold cyan]")

    try:
        with modal.enable_output(), onnx_convert_app.run():
            result = onnx_convert_model.remote(
                model_name=config.name,
                hf_source=config.hf_source,
                hf_target=config.hf_target,
                model_class=config.model_class,
                output_attr=config.output_attr,
                force=force,
            )

        status = result.get("status", "unknown")
        if status == "skipped":
            console.print(
                f"[yellow]Bo qua {config.name} — repo da ton tai tren HF Hub. "
                f"Dung --force de ghi de.[/yellow]"
            )
        elif status == "success":
            files_count = result.get("files_count", 0)
            total_size = result.get("total_size_mb", 0)
            url = result.get("url", "")
            variants = result.get("variants", {})
            console.print(
                f"[green]ONNX Convert {config.name}: THANH CONG "
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
            console.print(
                f"[red]ONNX Convert {config.name}: trạng thái không xác định — {result}[/red]"
            )
            raise typer.Exit(code=1) from None

    except modal.exception.AuthError:
        console.print("[red]Lỗi: Chưa xác thực Modal. Chạy `modal token set` trước.[/red]")
        raise typer.Exit(code=1) from None
    except Exception as e:
        console.print(f"[red]ONNX Convert {config.name}: THẤT BẠI — {e}[/red]")
        import traceback

        traceback.print_exc()
        raise typer.Exit(code=1) from None


@app.command("list")
def list_onnx_models() -> None:
    """Liệt kê tất cả ONNX models có thể convert."""
    table = Table(title="ONNX Model Registry")
    table.add_column("Tên", style="cyan")
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
