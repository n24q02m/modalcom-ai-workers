"""CLI entry point: python -m ai_workers.cli <command>"""


import typer

from ai_workers.cli.deploy import app as deploy_app
from ai_workers.cli.gguf_convert import app as gguf_convert_app
from ai_workers.cli.onnx_convert import app as onnx_convert_app

app = typer.Typer(
    name="ai-workers",
    help="AI Workers CLI — deploy models to Modal.com",
    no_args_is_help=True,
)

app.add_typer(
    onnx_convert_app,
    name="onnx-convert",
    help="Convert models to ONNX (INT8 + Q4F16) on Modal CPU → push to HF Hub",
)
app.add_typer(
    gguf_convert_app,
    name="gguf-convert",
    help="Convert models to GGUF Q4_K_M on Modal CPU → push to HF Hub",
)
app.add_typer(deploy_app, name="deploy", help="Deploy workers to Modal.com")

if __name__ == "__main__":
    app()
