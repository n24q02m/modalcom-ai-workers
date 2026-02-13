"""CLI entry point: python -m ai_workers.cli <command>"""

from __future__ import annotations

import typer

from ai_workers.cli.convert import app as convert_app
from ai_workers.cli.deploy import app as deploy_app
from ai_workers.cli.onnx_convert import app as onnx_convert_app
from ai_workers.cli.upload import app as upload_app

app = typer.Typer(
    name="ai-workers",
    help="AI Workers CLI — convert, deploy models lên Modal.com",
    no_args_is_help=True,
)

app.add_typer(
    convert_app,
    name="convert",
    help="[DEPRECATED] Convert models trên Modal CPU → R2 (workers giờ tải trực tiếp từ HF Hub)",
)
app.add_typer(
    onnx_convert_app,
    name="onnx-convert",
    help="Convert models sang ONNX INT8 trên Modal CPU → push HF Hub",
)
app.add_typer(
    upload_app,
    name="upload",
    help="[DEPRECATED] Upload thủ công từ local → R2",
)
app.add_typer(deploy_app, name="deploy", help="Deploy workers lên Modal.com")

if __name__ == "__main__":
    app()
