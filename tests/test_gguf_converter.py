"""Tests for gguf_converter module: registry, _generate_gguf_model_card, GgufModelConfig."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# gguf_convert_model
# ---------------------------------------------------------------------------
import sys
from unittest.mock import MagicMock, patch

import pytest

from ai_workers.workers.gguf_converter import (
    gguf_convert_model,
)

# Loguru uses sysconfig.get_path which might have issues in some mocked envs.
# tests/conftest.py usually stubs loguru.
# Let's ensure loguru is stubbed or we just mock it out.
mock_loguru = MagicMock()
sys.modules["loguru"] = mock_loguru

# Mock huggingface_hub globally since it's imported locally in the function
mock_huggingface_hub = MagicMock()
sys.modules["huggingface_hub"] = mock_huggingface_hub


@patch("os.environ.get")
@patch("subprocess.run")
@patch("gc.collect")
def test_gguf_convert_model_convert_subprocess_fails(
    mock_gc_collect,
    mock_subprocess_run,
    mock_environ_get,
):
    mock_environ_get.return_value = "fake-token"

    # Configure the mocked list_repo_tree within the globally mocked module
    mock_huggingface_hub.list_repo_tree.side_effect = Exception("Repo does not exist yet")

    # Mock subprocess.run to return a non-zero exit code for convert_hf_to_gguf.py
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "Conversion failed"
    mock_subprocess_run.return_value = mock_result

    with pytest.raises(RuntimeError, match=r"convert_hf_to_gguf\.py failed with exit code 1"):
        if hasattr(gguf_convert_model, "get_raw_f"):
            func = gguf_convert_model.get_raw_f()
        else:
            func = gguf_convert_model

        func(
            model_name="test-model-gguf",
            hf_source="org/test-model",
            hf_target="org/test-model-GGUF",
            gguf_name="test-model",
        )


# ---------------------------------------------------------------------------
@patch("os.environ.get")
@patch("subprocess.run")
@patch("gc.collect")
@patch("pathlib.Path.unlink")
def test_gguf_convert_model_quantize_subprocess_fails(
    mock_path_unlink,
    mock_gc_collect,
    mock_subprocess_run,
    mock_environ_get,
):
    mock_environ_get.return_value = "fake-token"

    # Mock huggingface_hub globally since it's imported locally in the function
    import sys

    mock_huggingface_hub = MagicMock()
    mock_huggingface_hub.list_repo_tree.side_effect = Exception("Repo does not exist yet")
    sys.modules["huggingface_hub"] = mock_huggingface_hub

    # Create mock results for the two subprocess.run calls
    # Call 1: convert_hf_to_gguf.py (success)
    mock_result_1 = MagicMock()
    mock_result_1.returncode = 0

    # Call 2: llama-quantize (failure)
    mock_result_2 = MagicMock()
    mock_result_2.returncode = 1
    mock_result_2.stderr = "Quantization failed"

    mock_subprocess_run.side_effect = [mock_result_1, mock_result_2]

    # Patch Path.stat() inside the function since it tries to get file size
    with patch("pathlib.Path.stat") as mock_stat:
        mock_stat.return_value.st_size = 1024 * 1024 * 10  # 10MB fake size

        with pytest.raises(RuntimeError, match="llama-quantize failed with exit code 1"):
            if hasattr(gguf_convert_model, "get_raw_f"):
                func = gguf_convert_model.get_raw_f()
            else:
                func = gguf_convert_model

            func(
                model_name="test-model-gguf",
                hf_source="org/test-model",
                hf_target="org/test-model-GGUF",
                gguf_name="test-model",
            )


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
