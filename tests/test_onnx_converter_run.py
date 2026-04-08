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
        mock_api_inst = mock_hf_hub.HfApi.return_value

        mock_q4f16_model = MagicMock()
        mock_node = MagicMock()
        mock_node.op_type = "Cast"
        mock_attr = MagicMock()
        mock_attr.name = "to"
        mock_attr.i = 1  # TensorProto.FLOAT
        mock_attr.g = None  # No subgraph - prevent isinstance() on MagicMock
        mock_node.attribute = [mock_attr]
        mock_q4f16_model.graph.node = [mock_node]

        mock_onnx_conv_common.float16.convert_float_to_float16.return_value = mock_q4f16_model

        # API mocks

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


def test_onnx_wrapper_forward():
    import torch

    mock_model = MagicMock()
    mock_out = MagicMock()
    mock_model.return_value = mock_out

    from ai_workers.workers.onnx_converter import _OnnxWrapper

    wrapper = _OnnxWrapper(mock_model, "last_hidden_state")

    dummy_ids = torch.ones(1, 10)
    dummy_mask = torch.ones(1, 10)

    # We call forward directly because _MODULE_BASE might be 'object' in test env if torch was not there at import time
    result = wrapper.forward(dummy_ids, dummy_mask)

    mock_model.assert_called_once_with(input_ids=dummy_ids, attention_mask=dummy_mask)
    assert result == mock_out.last_hidden_state


def test_yes_no_wrapper_forward():
    import torch
    import torch.nn as nn

    # Ensure torch.nn.Linear exists for the test
    if not hasattr(nn, "Linear"):
        nn.Linear = MagicMock()

    mock_model = MagicMock()
    mock_inner_model = MagicMock()
    mock_model.model = mock_inner_model

    mock_lm_head = MagicMock()
    # (vocab=32000, hidden=1024)
    mock_lm_head.weight.data = torch.ones(32000, 1024)
    mock_model.lm_head = mock_lm_head

    from ai_workers.workers.onnx_converter import _YesNoWrapper

    with patch("torch.nn.Linear", return_value=MagicMock()) as mock_linear_cls:
        mock_linear_inst = mock_linear_cls.return_value
        wrapper = _YesNoWrapper(mock_model)

        dummy_ids = torch.ones(1, 10)
        dummy_mask = torch.ones(1, 10)

        mock_out = MagicMock()
        mock_out.last_hidden_state = torch.ones(1, 10, 1024)
        mock_inner_model.return_value = mock_out

        result = wrapper.forward(dummy_ids, dummy_mask)

        mock_inner_model.assert_called_once_with(input_ids=dummy_ids, attention_mask=dummy_mask)
        # last_hidden = out.last_hidden_state[:, -1, :]
        mock_linear_inst.assert_called_once()
        assert result == mock_linear_inst.return_value


def test_fix_cast_nodes_recursive():

    # Define a real class for isinstance check since onnx is mocked
    class MockGraphProto:
        pass

    # Update the mock to include GraphProto
    mock_onnx.GraphProto = MockGraphProto

    mock_graph = MagicMock()
    mock_node = MagicMock()
    mock_node.op_type = "Cast"

    mock_attr_to = MagicMock()
    mock_attr_to.name = "to"
    mock_attr_to.i = 1  # FLOAT

    mock_subgraph = MockGraphProto()
    mock_subgraph.node = []  # Stop recursion
    mock_attr_g = MagicMock()
    mock_attr_g.g = mock_subgraph

    mock_node.attribute = [mock_attr_to, mock_attr_g]
    mock_graph.node = [mock_node]

    from ai_workers.workers.onnx_converter import _fix_cast_nodes

    _fix_cast_nodes(mock_graph)

    assert mock_attr_to.i == 10  # FLOAT16


def test_onnx_convert_model_yesno_logits_and_external_data():
    import torch
    import transformers

    with (
        patch.dict(os.environ, {"HF_TOKEN": "test-token"}),
        patch("pathlib.Path.stat") as mock_stat,
        patch("pathlib.Path.unlink"),
        patch("pathlib.Path.rglob") as mock_rglob,
        patch("pathlib.Path.write_text"),
        patch("pathlib.Path.exists") as mock_exists,
    ):
        mock_hf_hub.repo_exists.return_value = False

        # Transformers mocks
        mock_tokenizer = MagicMock()
        transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.parameters.return_value = [torch.ones(10)]
        mock_model.lm_head.weight.data = torch.ones(32000, 1024)
        transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model

        mock_config = MagicMock()
        transformers.AutoConfig.from_pretrained.return_value = mock_config

        # ONNX mocks
        mock_stat.return_value.st_size = 100 * 1024 * 1024  # 100MB

        # Mock .onnx.data exists
        mock_exists.side_effect = lambda: True  # For all .exists() calls

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
        mock_q4f16_model.graph.node = []
        mock_onnx_conv_common.float16.convert_float_to_float16.return_value = mock_q4f16_model

        # API mocks

        from ai_workers.workers.onnx_converter import onnx_convert_model

        result = onnx_convert_model(
            model_name="test-reranker",
            hf_source="org/src",
            hf_target="org/tgt",
            model_class="AutoModelForCausalLM",
            output_attr="yesno_logits",
        )

        assert result["status"] == "success"
        assert result["model_name"] == "test-reranker"

        # Verify yesno_logits specific calls
        transformers.AutoModelForCausalLM.from_pretrained.assert_called_once()


def test_onnx_converter_import_error():
    import importlib
    import sys

    # Save original sys.modules
    orig_torch = sys.modules.get("torch")
    orig_torch_nn = sys.modules.get("torch.nn")

    try:
        # Simulate missing torch
        sys.modules["torch"] = None
        sys.modules["torch.nn"] = None

        # Reload the module to trigger the except block
        import ai_workers.workers.onnx_converter as onnx_mod

        importlib.reload(onnx_mod)

        # In the except block, _MODULE_BASE should be object
        assert onnx_mod._MODULE_BASE is object

    finally:
        # Restore sys.modules
        if orig_torch:
            sys.modules["torch"] = orig_torch
        else:
            del sys.modules["torch"]

        if orig_torch_nn:
            sys.modules["torch.nn"] = orig_torch_nn
        else:
            del sys.modules["torch.nn"]

        # Reload again to restore normal state for other tests
        import ai_workers.workers.onnx_converter as onnx_mod

        importlib.reload(onnx_mod)
