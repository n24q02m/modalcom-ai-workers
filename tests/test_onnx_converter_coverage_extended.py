from __future__ import annotations

import importlib
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Test _MODULE_BASE fallback
# ---------------------------------------------------------------------------


def test_module_base_fallback():
    # We must ensure it's in sys.modules to reload it
    import ai_workers.workers.onnx_converter

    with patch.dict("sys.modules", {"torch": None, "torch.nn": None}):
        importlib.reload(ai_workers.workers.onnx_converter)
        assert ai_workers.workers.onnx_converter._MODULE_BASE is object

    # Reload back to normal
    importlib.reload(ai_workers.workers.onnx_converter)


# ---------------------------------------------------------------------------
# Extended Coverage Tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mock_modules():
    mock_onnx = MagicMock()
    mock_onnx.TensorProto.FLOAT = 1
    mock_onnx.TensorProto.FLOAT16 = 10
    mock_onnx.GraphProto = MagicMock

    mock_ort_quant = MagicMock()
    mock_ort_quant.QuantType.QInt8 = "QInt8"

    mock_nbits = MagicMock()
    mock_nbits_quantizer_cls = MagicMock()
    mock_nbits.MatMulNBitsQuantizer = mock_nbits_quantizer_cls

    mock_hf_hub = MagicMock()
    mock_hf_hub.HfApi = MagicMock()
    mock_hf_hub.repo_exists = MagicMock()

    mock_onnx_conv_common = MagicMock()

    with patch.dict(
        "sys.modules",
        {
            "onnx": mock_onnx,
            "onnxruntime": MagicMock(),
            "onnxruntime.quantization": mock_ort_quant,
            "onnxruntime.quantization.matmul_nbits_quantizer": mock_nbits,
            "huggingface_hub": mock_hf_hub,
            "onnxconverter_common": mock_onnx_conv_common,
        },
    ):
        yield {
            "onnx": mock_onnx,
            "ort_quant": mock_ort_quant,
            "nbits": mock_nbits,
            "hf_hub": mock_hf_hub,
            "onnx_conv_common": mock_onnx_conv_common,
            "nbits_quantizer_cls": mock_nbits_quantizer_cls,
        }


def test_fix_cast_nodes_recursion(mock_modules):
    from ai_workers.workers.onnx_converter import _fix_cast_nodes

    mock_onnx = mock_modules["onnx"]

    mock_graph = MagicMock()
    mock_node = MagicMock()
    mock_node.op_type = "Cast"

    mock_attr_to = MagicMock()
    mock_attr_to.name = "to"
    mock_attr_to.i = 1  # FLOAT

    mock_subgraph = MagicMock()
    mock_subgraph.__class__ = mock_onnx.GraphProto

    mock_attr_g = MagicMock()
    mock_attr_g.g = mock_subgraph

    mock_node.attribute = [mock_attr_to, mock_attr_g]
    mock_graph.node = [mock_node]

    mock_subnode = MagicMock()
    mock_subnode.op_type = "Cast"
    mock_subattr_to = MagicMock()
    mock_subattr_to.name = "to"
    mock_subattr_to.i = 1  # FLOAT
    mock_subnode.attribute = [mock_subattr_to]
    mock_subgraph.node = [mock_subnode]

    with patch("ai_workers.workers.onnx_converter.isinstance", return_value=True):
        _fix_cast_nodes(mock_graph)

    assert mock_attr_to.i == 10  # FLOAT16
    assert mock_subattr_to.i == 10  # FLOAT16


def test_yes_no_wrapper():
    import torch
    import torch.nn as nn

    from ai_workers.workers.onnx_converter import _YesNoWrapper

    # Ensure nn.Linear exists (it's a stub in conftest)
    if not hasattr(nn, "Linear"):
        nn.Linear = MagicMock(return_value=MagicMock())

    mock_inner = MagicMock()
    mock_inner.model = MagicMock()
    mock_inner.lm_head.weight.data = torch.randn(10000, 128)

    wrapper = _YesNoWrapper(mock_inner)

    dummy_ids = torch.zeros((1, 10), dtype=torch.long)
    dummy_mask = torch.ones((1, 10), dtype=torch.long)

    mock_out = MagicMock()
    mock_out.last_hidden_state = torch.randn(1, 10, 128)
    mock_inner.model.return_value = mock_out

    mock_logits = torch.randn(1, 2)
    wrapper.yes_no_head.return_value = mock_logits

    logits = wrapper.forward(dummy_ids, dummy_mask)
    assert logits is mock_logits


def test_onnx_wrapper():
    import torch

    from ai_workers.workers.onnx_converter import _OnnxWrapper

    mock_inner = MagicMock()
    mock_hidden = torch.randn(1, 10, 128)
    mock_inner.return_value = MagicMock(last_hidden_state=mock_hidden)

    wrapper = _OnnxWrapper(mock_inner, "last_hidden_state")

    dummy_ids = torch.zeros((1, 10), dtype=torch.long)
    dummy_mask = torch.ones((1, 10), dtype=torch.long)

    out = wrapper.forward(dummy_ids, dummy_mask)
    assert out is mock_hidden


def test_onnx_convert_model_yesno_and_external_data(mock_modules):
    import torch
    import torch.nn as nn
    import transformers

    from ai_workers.workers.onnx_converter import onnx_convert_model

    # Ensure nn.Linear exists
    if not hasattr(nn, "Linear"):
        nn.Linear = MagicMock(return_value=MagicMock())

    mock_hf_hub = mock_modules["hf_hub"]
    mock_nbits_quantizer_cls = mock_modules["nbits_quantizer_cls"]
    mock_onnx_conv_common = mock_modules["onnx_conv_common"]

    with (
        patch.dict(os.environ, {"HF_TOKEN": "test-token"}),
        patch.object(
            Path, "stat", side_effect=lambda *args, **kwargs: MagicMock(st_size=100 * 1024 * 1024)
        ),
        patch.object(Path, "unlink"),
        patch.object(Path, "exists", side_effect=lambda *args, **kwargs: True) as mock_exists,
        patch.object(Path, "rglob") as mock_rglob,
        patch("pathlib.Path.write_text"),
    ):
        mock_hf_hub.repo_exists.return_value = False

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros((1, 5)),
            "attention_mask": torch.ones((1, 5)),
        }
        transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.parameters.return_value = [torch.ones(10)]
        mock_model.lm_head.weight.data = torch.randn(10000, 128)
        transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model

        mock_config = MagicMock()
        transformers.AutoConfig.from_pretrained.return_value = mock_config

        mock_file = MagicMock()
        mock_file.is_file.return_value = True
        mock_file.stat.return_value.st_size = 50 * 1024 * 1024
        mock_file.relative_to.return_value = Path("onnx/model_quantized.onnx")
        mock_rglob.return_value = [mock_file]

        mock_quantizer_inst = mock_nbits_quantizer_cls.return_value
        mock_quantizer_inst.model.model = MagicMock()

        mock_q4f16_model = MagicMock()
        mock_q4f16_model.graph.node = []
        mock_onnx_conv_common.float16.convert_float_to_float16.return_value = mock_q4f16_model

        result = onnx_convert_model(
            model_name="test-model",
            hf_source="org/src",
            hf_target="org/tgt",
            model_class="AutoModelForCausalLM",
            output_attr="yesno_logits",
        )

        assert result["status"] == "success"
        assert mock_exists.called
