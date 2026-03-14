"""Tests for the convert CLI commands."""

from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from ai_workers.cli.convert import app
from ai_workers.common.config import (
    GPU,
    MODEL_REGISTRY,
    ModelClassType,
    ModelConfig,
    Precision,
    ServingEngine,
    Task,
    Tier,
)

runner = CliRunner()


def test_list_available() -> None:
    """Test the list command outputs the correct table."""
    # Create sample models for testing
    mock_registry = {
        "test-model-1": ModelConfig(
            name="test-model-1",
            hf_id="Test/Model-1",
            task=Task.EMBEDDING,
            tier=Tier.LIGHT,
            precision=Precision.FP16,
            gpu=GPU.T4,
            serving_engine=ServingEngine.CUSTOM_FASTAPI,
            model_class=ModelClassType.AUTO_MODEL,
        ),
        "test-model-2": ModelConfig(
            name="test-model-2",
            hf_id="Test/Model-2",
            task=Task.RERANKER_LLM,
            tier=Tier.HEAVY,
            precision=Precision.FP16,
            gpu=GPU.A10G,
            serving_engine=ServingEngine.CUSTOM_FASTAPI,
            model_class=ModelClassType.CAUSAL_LM,
        ),
    }

    # Set COLUMNS to ensure the table isn't wrapped too aggressively
    with patch.dict(MODEL_REGISTRY, mock_registry, clear=True):
        result = runner.invoke(app, ["list"], env={"COLUMNS": "200"})

    assert result.exit_code == 0
    # Check table headers are present
    assert "Model Registry" in result.stdout
    assert "Name" in result.stdout
    assert "HuggingFace ID" in result.stdout
    assert "Task" in result.stdout
    assert "Tier" in result.stdout
    assert "Precision" in result.stdout
    assert "GPU" in result.stdout

    # Check model data is present
    assert "test-model-1" in result.stdout
    assert "Test/Model-1" in result.stdout
    assert Task.EMBEDDING.value in result.stdout
    assert Tier.LIGHT.value in result.stdout
    assert Precision.FP16.value in result.stdout
    assert GPU.T4.value in result.stdout

    assert "test-model-2" in result.stdout
    assert "Test/Model-2" in result.stdout
    assert Task.RERANKER_LLM.value in result.stdout
    assert Tier.HEAVY.value in result.stdout
    assert GPU.A10G.value in result.stdout


def test_list_available_empty() -> None:
    """Test the list command handles an empty registry."""
    with patch.dict(MODEL_REGISTRY, {}, clear=True):
        result = runner.invoke(app, ["list"], env={"COLUMNS": "200"})

    assert result.exit_code == 0
    assert "Model Registry" in result.stdout
    # Should show headers but no rows
    assert "Name" in result.stdout
