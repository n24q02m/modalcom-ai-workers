"""CLI entry point: python -m ai_workers.cli <command>"""

from __future__ import annotations

import typer

from ai_workers.cli.convert import app as convert_app
from ai_workers.cli.deploy import app as deploy_app
from ai_workers.cli.gguf_convert import app as gguf_convert_app
from ai_workers.cli.onnx_convert import app as onnx_convert_app
from ai_workers.cli.upload import app as upload_app

app = typer.Typer(
    name="ai-workers",
    help="AI Workers CLI — convert, deploy models len Modal.com",
    no_args_is_help=True,
)

app.add_typer(
    convert_app,
    name="convert",
    help="[DEPRECATED] Convert models tren Modal CPU -> R2 (workers gio tai truc tiep tu HF Hub)",
)
app.add_typer(
    onnx_convert_app,
    name="onnx-convert",
    help="Convert models sang ONNX (INT8 + Q4F16) tren Modal CPU -> push HF Hub",
)
app.add_typer(
    gguf_convert_app,
    name="gguf-convert",
    help="Convert models sang GGUF Q4_K_M tren Modal CPU -> push HF Hub",
)
app.add_typer(
    upload_app,
    name="upload",
    help="[DEPRECATED] Upload thu cong tu local -> R2",
)
app.add_typer(deploy_app, name="deploy", help="Deploy workers len Modal.com")

if __name__ == "__main__":
    app()
