"""[DEPRECATED] Convert HuggingFace models sang SafeTensors va ghi len R2.

DEPRECATED: Workers gio tai model truc tiep tu HuggingFace Hub qua Xet protocol.
Khong can convert rieng nua. Lenh nay chi giu lai de backward compatibility.

Neu can convert sang ONNX INT8, dung lenh `onnx-convert` thay the.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from ai_workers.common.config import (
    MODEL_REGISTRY,
    get_model,
    list_models,
)

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.callback(invoke_without_command=True)
def convert(
    model: str = typer.Argument(
        None,
        help="Tên model trong registry (vd: 'qwen3-embedding-0.6b'), 'all', hoặc 'list'",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Ghi đè nếu model đã tồn tại trên R2",
    ),
) -> None:
    """[DEPRECATED] Convert model HuggingFace sang SafeTensors tren Modal CPU.

    Workers gio tai model truc tiep tu HuggingFace Hub. Khong can convert rieng.
    """
    console.print(
        "[yellow bold]DEPRECATED: Workers gio tai model truc tiep tu HuggingFace Hub "
        "qua Xet protocol. Khong can convert rieng nua.[/yellow bold]\n"
    )

    if model is None:
        console.print(
            "[yellow]Cần chỉ định model, 'all', hoặc 'list'. Dùng --help để xem hướng dẫn.[/yellow]"
        )
        raise typer.Exit(code=1)

    if model == "list":
        list_available()
        return

    if model == "all":
        models = list_models()
        console.print(f"[bold]Converting {len(models)} models trên Modal CPU...[/bold]")
        failed: list[str] = []
        for m in models:
            try:
                _convert_remote(m.name, force=force)
            except (SystemExit, Exception):
                failed.append(m.name)
        if failed:
            console.print(
                f"\n[red bold]{len(failed)} model(s) thất bại: {', '.join(failed)}[/red bold]"
            )
            raise typer.Exit(code=1)
        console.print(f"\n[green bold]Tất cả {len(models)} models convert thành công![/green bold]")
        return

    _convert_remote(model, force=force)


def _convert_remote(model_name: str, *, force: bool = False) -> None:
    """Gọi Modal remote function để convert một model."""
    try:
        config = get_model(model_name)
    except KeyError as e:
        console.print(f"[red]Lỗi: {e}[/red]")
        raise typer.Exit(code=1) from None

    console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
    console.print(f"[bold]Converting: {config.name}[/bold]")
    console.print(f"  HuggingFace ID: {config.hf_id}")
    console.print(f"  Precision: {config.precision.value}")
    console.print(f"  Model Class: {config.model_class.value}")
    console.print("  Chạy trên: Modal CPU (32GB RAM)")
    console.print("  Đầu ra: R2 bucket (CloudBucketMount)")
    console.print(f"[bold cyan]{'=' * 60}[/bold cyan]")

    try:
        import modal

        from ai_workers.workers.converter import convert_app, convert_model

        with modal.enable_output(), convert_app.run():
            result = convert_model.remote(
                model_name=config.name,
                hf_id=config.hf_id,
                precision=config.precision.value,
                model_class=config.model_class.value,
                task=config.task.value,
                trust_remote_code=config.trust_remote_code,
                extra_load_kwargs=config.extra_load_kwargs,
                force=force,
            )

        status = result.get("status", "unknown")
        if status == "skipped":
            console.print(
                f"[yellow]Bỏ qua {config.name} — đã tồn tại trên R2. "
                f"Dùng --force để ghi đè.[/yellow]"
            )
        elif status == "success":
            files_count = result.get("files_count", 0)
            total_size = result.get("total_size_mb", 0)
            console.print(
                f"[green]Convert {config.name}: THÀNH CÔNG "
                f"({files_count} files, {total_size:.2f} MB)[/green]"
            )
        else:
            console.print(f"[red]Convert {config.name}: trạng thái không xác định — {result}[/red]")
            raise typer.Exit(code=1) from None

    except modal.exception.AuthError:
        console.print("[red]Lỗi: Chưa xác thực Modal. Chạy `modal token set` trước.[/red]")
        raise typer.Exit(code=1) from None
    except Exception as e:
        console.print(f"[red]Convert {config.name}: THẤT BẠI — {e}[/red]")
        import traceback

        traceback.print_exc()
        raise typer.Exit(code=1) from None


@app.command("list")
def list_available() -> None:
    """Liệt kê tất cả models trong registry."""
    table = Table(title="Model Registry")
    table.add_column("Tên", style="cyan")
    table.add_column("HuggingFace ID", style="dim")
    table.add_column("Task")
    table.add_column("Tier")
    table.add_column("Precision")
    table.add_column("GPU (deploy)")

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
