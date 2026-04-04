from __future__ import annotations

from unittest.mock import patch

import typer
from typer.testing import CliRunner

from ai_workers.cli.deploy import app
from ai_workers.common.config import (
    ModelConfig,
    Task,
    Tier,
)

runner = CliRunner()


def test_deploy_all_failure_reporting():
    """Test that deploy --all correctly aggregates and reports failures."""

    # Mock models
    m1 = ModelConfig(
        name="m1",
        hf_id="h1",
        task=Task.EMBEDDING,
        tier=Tier.LIGHT,
        worker_module="mod1",
        modal_app_var="app1",
    )
    m2 = ModelConfig(
        name="m2",
        hf_id="h2",
        task=Task.EMBEDDING,
        tier=Tier.HEAVY,
        worker_module="mod1",
        modal_app_var="app1",
    )
    m3 = ModelConfig(
        name="m3",
        hf_id="h3",
        task=Task.RERANKER_LLM,
        tier=Tier.HEAVY,
        worker_module="mod2",
        modal_app_var="app2",
    )

    mock_models = [m1, m2, m3]

    with (
        patch("ai_workers.cli.deploy.list_models", return_value=mock_models),
        patch("ai_workers.cli.deploy._deploy_app") as mock_deploy_app,
    ):
        # mod1/app1 fails, mod2/app2 succeeds
        def side_effect(module, app_var, dry_run=False):
            if module == "mod1" and app_var == "app1":
                raise typer.Exit(code=1)
            return None

        mock_deploy_app.side_effect = side_effect

        result = runner.invoke(app, ["--all"])

        assert result.exit_code == 1
        # The code joins failures with ", "
        # failures.extend(names) where names = ["m1", "m2"]
        assert "Deploy FAILED: m1, m2" in result.output

        # Verify both apps were attempted (even if one failed)
        assert mock_deploy_app.call_count == 2
