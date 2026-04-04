# ruff: noqa: E402
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Mock modules that are imported locally inside the function
mock_onnx = MagicMock()
mock_onnx.TensorProto.FLOAT = 1
mock_onnx.TensorProto.FLOAT16 = 10

mock_ort_quant = MagicMock()
mock_ort_quant.QuantType.QInt8 = "QInt8"
mock_ort_quant.quantize_dynamic = MagicMock()

mock_nbits = MagicMock()
mock_nbits_quantizer_cls = MagicMock()
mock_nbits.MatMulNBitsQuantizer = mock_nbits_quantizer_cls

mock_hf_hub = MagicMock()
mock_hf_hub.HfApi = MagicMock()
mock_hf_hub.repo_exists = MagicMock()

mock_onnx_conv_common = MagicMock()

# Global patch to ensure modules are available throughout the test session
sys.modules["onnx"] = mock_onnx
sys.modules["onnxruntime"] = MagicMock()
sys.modules["onnxruntime.quantization"] = mock_ort_quant
sys.modules["onnxruntime.quantization.matmul_nbits_quantizer"] = mock_nbits
sys.modules["huggingface_hub"] = mock_hf_hub
sys.modules["onnxconverter_common"] = mock_onnx_conv_common

from ai_workers.workers.onnx_converter import onnx_convert_model


@pytest.fixture(autouse=True)
def reset_mocks():
    mock_onnx.reset_mock()
    mock_ort_quant.reset_mock()
    mock_nbits.reset_mock()
    mock_hf_hub.reset_mock()
    mock_onnx_conv_common.reset_mock()

    import transformers

    # Reset transformers mocks which are in sys.modules from conftest
    transformers.AutoTokenizer.from_pretrained.reset_mock()
    transformers.AutoModel.from_pretrained.reset_mock()
    transformers.AutoModelForCausalLM.from_pretrained.reset_mock()
    transformers.AutoConfig.from_pretrained.reset_mock()

    yield


# ---------------------------------------------------------------------------
# Infrastructure Verification
# ---------------------------------------------------------------------------


def test_infrastructure_is_mocked():
    assert "onnx" in sys.modules
    assert "huggingface_hub" in sys.modules
    assert "transformers" in sys.modules  # From conftest
    assert "torch" in sys.modules  # From conftest


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_onnx_convert_model_missing_token():
    with patch.dict(os.environ, {}, clear=True):
        if "HF_TOKEN" in os.environ:
            del os.environ["HF_TOKEN"]
        with pytest.raises(ValueError, match="HF_TOKEN is not set"):
            onnx_convert_model(
                model_name="test",
                hf_source="src",
                hf_target="tgt",
                model_class="AutoModel",
                output_attr="last_hidden_state",
            )


def test_onnx_convert_model_already_exists():
    with patch.dict(os.environ, {"HF_TOKEN": "test-token"}):
        mock_hf_hub.repo_exists.return_value = True

        result = onnx_convert_model(
            model_name="test",
            hf_source="src",
            hf_target="tgt",
            model_class="AutoModel",
            output_attr="last_hidden_state",
            force=False,
        )

        assert result["status"] == "skipped"
        assert result["reason"] == "already_exists"
        mock_hf_hub.repo_exists.assert_called_once_with("tgt", token="test-token")


def test_onnx_convert_model_invalid_class():
    with patch.dict(os.environ, {"HF_TOKEN": "test-token"}):
        mock_hf_hub.repo_exists.return_value = False

        with pytest.raises(ValueError, match="Model class 'InvalidClass' is invalid"):
            onnx_convert_model(
                model_name="test",
                hf_source="src",
                hf_target="tgt",
                model_class="InvalidClass",
                output_attr="last_hidden_state",
            )


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


def test_onnx_convert_model_success():
    import torch
    import transformers

    with (
        patch.dict(os.environ, {"HF_TOKEN": "test-token"}),
        patch("pathlib.Path.stat") as mock_stat,
        patch("pathlib.Path.unlink"),
        patch("pathlib.Path.rglob") as mock_rglob,
        patch("pathlib.Path.write_text"),
    ):
        mock_hf_hub.repo_exists.return_value = False

        # Transformers mocks
        mock_tokenizer = MagicMock()
        transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.parameters.return_value = [torch.ones(10)]
        transformers.AutoModel.from_pretrained.return_value = mock_model

        mock_config = MagicMock()
        transformers.AutoConfig.from_pretrained.return_value = mock_config

        # ONNX mocks
        mock_stat.return_value.st_size = 100 * 1024 * 1024  # 100MB

        # Mock file iteration for stats
        mock_file = MagicMock()
        mock_file.is_file.return_value = True
        mock_file.stat.return_value.st_size = 50 * 1024 * 1024
        mock_file.relative_to.return_value = Path("onnx/model_quantized.onnx")
        mock_rglob.return_value = [mock_file]

        # Quantizer mocks
        mock_quantizer_inst = mock_nbits_quantizer_cls.return_value
        mock_quantizer_inst.model.model = MagicMock()

        mock_q4f16_model = MagicMock()
        mock_node = MagicMock()
        mock_node.op_type = "Cast"
        mock_attr = MagicMock()
        mock_attr.name = "to"
        mock_attr.i = 1  # TensorProto.FLOAT
        mock_node.attribute = [mock_attr]
        mock_q4f16_model.graph.node = [mock_node]

        mock_onnx_conv_common.float16.convert_float_to_float16.return_value = mock_q4f16_model

        # API mocks
        mock_api_inst = mock_hf_hub.HfApi.return_value

        result = onnx_convert_model(
            model_name="test-model",
            hf_source="org/src",
            hf_target="org/tgt",
            model_class="AutoModel",
            output_attr="last_hidden_state",
        )

        assert result["status"] == "success"
        assert result["model_name"] == "test-model"
        assert result["hf_target"] == "org/tgt"
        assert "variants" in result

        # Verify calls
        transformers.AutoTokenizer.from_pretrained.assert_called_once()
        transformers.AutoModel.from_pretrained.assert_called_once()
        mock_ort_quant.quantize_dynamic.assert_called_once()
        mock_nbits_quantizer_cls.assert_called_once()
        mock_onnx_conv_common.float16.convert_float_to_float16.assert_called_once()
        mock_onnx.save.assert_called_once()
        mock_api_inst.create_repo.assert_called_once()
        mock_api_inst.upload_folder.assert_called_once()
