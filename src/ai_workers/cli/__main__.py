"""CLI entry point: python -m ai_workers.cli <command>"""

from __future__ import annotations

import typer  # type: ignore

from ai_workers.cli.convert import app as convert_app
from ai_workers.cli.deploy import app as deploy_app
from ai_workers.cli.upload import app as upload_app

app = typer.Typer(
    name="ai-workers",
    help="AI Workers CLI — convert, upload, and deploy models to Modal.com",
    no_args_is_help=True,
)

app.add_typer(convert_app, name="convert", help="Convert HuggingFace models to SafeTensors")
app.add_typer(upload_app, name="upload", help="Upload converted models to CF R2")
app.add_typer(deploy_app, name="deploy", help="Deploy workers to Modal.com")

if __name__ == "__main__":
    app()
