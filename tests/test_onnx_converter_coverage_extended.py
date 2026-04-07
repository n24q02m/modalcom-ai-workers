# ruff: noqa: N806
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# Helper to create/reset mocks
def get_mocks():
    mock_onnx = MagicMock()
    mock_onnx.TensorProto.FLOAT = 1
    mock_onnx.TensorProto.FLOAT16 = 10
    mock_onnx.GraphProto = type("GraphProto", (), {})

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

    return {
        "onnx": mock_onnx,
        "onnxruntime.quantization": mock_ort_quant,
        "onnxruntime.quantization.matmul_nbits_quantizer": mock_nbits,
        "huggingface_hub": mock_hf_hub,
        "onnxconverter_common": mock_onnx_conv_common,
    }


@pytest.fixture
def clean_mocks():
    mocks = get_mocks()
    with patch.dict(
        "sys.modules",
        {
            "onnx": mocks["onnx"],
            "onnxruntime": MagicMock(),
            "onnxruntime.quantization": mocks["onnxruntime.quantization"],
            "onnxruntime.quantization.matmul_nbits_quantizer": mocks[
                "onnxruntime.quantization.matmul_nbits_quantizer"
            ],
            "huggingface_hub": mocks["huggingface_hub"],
            "onnxconverter_common": mocks["onnxconverter_common"],
        },
    ):
        # Manually inject Linear if not present
        import torch.nn as nn

        if not hasattr(nn, "Linear"):
            nn.Linear = MagicMock()

        from ai_workers.workers.onnx_converter import (
            _fix_cast_nodes,
            _OnnxWrapper,
            _YesNoWrapper,
            onnx_convert_model,
        )

        yield mocks, _OnnxWrapper, _YesNoWrapper, _fix_cast_nodes, onnx_convert_model


def test_wrappers_forward(clean_mocks):
    _mocks, _OnnxWrapper, _YesNoWrapper, _fix_cast_nodes, _ = clean_mocks
    import torch

    # Test _OnnxWrapper
    inner_model = MagicMock()
    inner_model.return_value = MagicMock(attr="output")
    wrapper = _OnnxWrapper(inner_model, "attr")

    ids = torch.ones((1, 10))
    mask = torch.ones((1, 10))
    res = wrapper.forward(ids, mask)

    assert res == "output"
    inner_model.assert_called_once_with(input_ids=ids, attention_mask=mask)

    # Test _YesNoWrapper
    inner_model_yn = MagicMock()
    inner_model_yn.model.return_value = MagicMock(last_hidden_state=torch.ones((1, 1, 128)))
    inner_model_yn.lm_head.weight.data = torch.randn((10000, 128))

    wrapper_yn = _YesNoWrapper(inner_model_yn)
    res_yn = wrapper_yn.forward(ids, mask)

    assert isinstance(res_yn, MagicMock)
    inner_model_yn.model.assert_called_once_with(input_ids=ids, attention_mask=mask)


def test_fix_cast_nodes_recursion(clean_mocks):
    mocks, _, _, _fix_cast_nodes, _ = clean_mocks
    mock_graph = MagicMock()

    # Node 1: Cast to FLOAT
    node1 = MagicMock()
    node1.op_type = "Cast"
    attr1 = MagicMock()
    attr1.name = "to"
    attr1.i = 1  # FLOAT
    node1.attribute = [attr1]

    # Node 2: If node with subgraph
    node2 = MagicMock()
    node2.op_type = "If"
    attr2 = MagicMock()
    attr2.name = "then_branch"
    attr2.g = MagicMock()
    attr2.g.__class__ = mocks[
        "onnx"
    ].GraphProto  # Ensure isinstance(attr.g, onnx.GraphProto) is True

    subgraph = attr2.g
    subnode = MagicMock()
    subnode.op_type = "Cast"
    subattr = MagicMock()
    subattr.name = "to"
    subattr.i = 1  # FLOAT
    subnode.attribute = [subattr]
    subgraph.node = [subnode]

    node2.attribute = [attr2]

    mock_graph.node = [node1, node2]

    _fix_cast_nodes(mock_graph)

    assert attr1.i == 10  # FLOAT16
    assert subattr.i == 10  # FLOAT16


def test_onnx_convert_model_yesno_logits(clean_mocks):
    mocks, _, _, _, onnx_convert_model = clean_mocks
    import torch
    import transformers

    with (
        patch.dict(os.environ, {"HF_TOKEN": "test-token"}),
        patch("pathlib.Path.stat") as mock_stat,
        patch("pathlib.Path.unlink"),
        patch("pathlib.Path.rglob") as mock_rglob,
        patch("pathlib.Path.write_text"),
    ):
        mocks["huggingface_hub"].repo_exists.return_value = False

        # Transformers mocks
        mock_tokenizer = MagicMock()
        transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.parameters.return_value = [torch.ones(10)]
        mock_model.lm_head.weight.data = torch.randn((10000, 128))
        transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model

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
        mock_nbits_quantizer_cls = mocks[
            "onnxruntime.quantization.matmul_nbits_quantizer"
        ].MatMulNBitsQuantizer
        mock_quantizer_inst = mock_nbits_quantizer_cls.return_value
        mock_quantizer_inst.model.model = MagicMock()
        mock_q4f16_model = MagicMock()
        mock_q4f16_model.graph.node = []
        mocks[
            "onnxconverter_common"
        ].float16.convert_float_to_float16.return_value = mock_q4f16_model

        result = onnx_convert_model(
            model_name="test-model",
            hf_source="org/src",
            hf_target="org/tgt",
            model_class="AutoModelForCausalLM",
            output_attr="yesno_logits",
        )

        assert result["status"] == "success"
        transformers.AutoModelForCausalLM.from_pretrained.assert_called_once()


def test_onnx_convert_model_with_external_data(clean_mocks):
    mocks, _, _, _, onnx_convert_model = clean_mocks
    import torch
    import transformers

    with (
        patch.dict(os.environ, {"HF_TOKEN": "test-token"}),
        patch("pathlib.Path.stat") as mock_stat,
        patch("pathlib.Path.unlink") as mock_unlink,
        patch("pathlib.Path.rglob") as mock_rglob,
        patch("pathlib.Path.write_text"),
    ):
        mocks["huggingface_hub"].repo_exists.return_value = False

        # Transformers mocks
        mock_tokenizer = MagicMock()
        transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.parameters.return_value = [torch.ones(10)]
        transformers.AutoModel.from_pretrained.return_value = mock_model

        # Config mock with 100% coverage for transformers.AutoConfig.from_pretrained
        mock_config = MagicMock()
        transformers.AutoConfig.from_pretrained.return_value = mock_config

        # ONNX mocks
        mock_stat.return_value.st_size = 100 * 1024 * 1024  # 100MB

        with patch("pathlib.Path.exists", return_value=True):
            # Mock file iteration for stats
            mock_file = MagicMock()
            mock_file.is_file.return_value = True
            mock_file.stat.return_value.st_size = 50 * 1024 * 1024
            mock_file.relative_to.return_value = Path("onnx/model_quantized.onnx")
            mock_rglob.return_value = [mock_file]

            # Quantizer mocks
            mock_nbits_quantizer_cls = mocks[
                "onnxruntime.quantization.matmul_nbits_quantizer"
            ].MatMulNBitsQuantizer
            mock_quantizer_inst = mock_nbits_quantizer_cls.return_value
            mock_quantizer_inst.model.model = MagicMock()
            mock_q4f16_model = MagicMock()
            mock_q4f16_model.graph.node = []
            mocks[
                "onnxconverter_common"
            ].float16.convert_float_to_float16.return_value = mock_q4f16_model

            result = onnx_convert_model(
                model_name="test-model",
                hf_source="org/src",
                hf_target="org/tgt",
                model_class="AutoModel",
                output_attr="last_hidden_state",
            )

        assert result["status"] == "success"
        # Verify unlink was called for both .onnx and .onnx.data
        assert mock_unlink.call_count >= 2


def test_onnx_convert_model_security_and_errors(clean_mocks):
    _, _, _, _, onnx_convert_model = clean_mocks

    with patch.dict(os.environ, {"HF_TOKEN": "test-token"}):
        # 1. Untrusted organization
        with pytest.raises(ValueError, match="Untrusted organization 'malicious-org'"):
            onnx_convert_model(
                model_name="test",
                hf_source="malicious-org/evil",
                hf_target="mine/evil",
                model_class="AutoModel",
                output_attr="logits",
                trust_remote_code=True,
            )

        # 2. Missing token
        with patch.dict(os.environ, {}, clear=True):
            if "HF_TOKEN" in os.environ:
                del os.environ["HF_TOKEN"]
            with pytest.raises(ValueError, match="HF_TOKEN is not set"):
                onnx_convert_model(
                    model_name="test",
                    hf_source="src",
                    hf_target="tgt",
                    model_class="AutoModel",
                    output_attr="logits",
                )
