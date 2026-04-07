import pytest

from ai_workers.workers.onnx_converter import onnx_convert_model


def test_onnx_convert_invalid_hf_source_path_traversal():
    with pytest.raises(ValueError, match="Invalid hf_source"):
        onnx_convert_model(
            model_name="test",
            hf_source="Qwen/../../etc/passwd",
            hf_target="target/repo",
            model_class="AutoModel",
            output_attr="last_hidden_state",
        )


def test_onnx_convert_invalid_hf_target_path_traversal():
    with pytest.raises(ValueError, match="Invalid hf_target"):
        onnx_convert_model(
            model_name="test",
            hf_source="Qwen/Model",
            hf_target="target/../../tmp/evil",
            model_class="AutoModel",
            output_attr="last_hidden_state",
        )


def test_onnx_convert_invalid_output_attr():
    with pytest.raises(ValueError, match="Invalid output_attr"):
        onnx_convert_model(
            model_name="test",
            hf_source="Qwen/Model",
            hf_target="target/repo",
            model_class="AutoModel",
            output_attr="malicious_attr",
        )


def test_onnx_convert_untrusted_org_with_remote_code():
    with pytest.raises(ValueError, match="Untrusted organization"):
        onnx_convert_model(
            model_name="test",
            hf_source="attacker/malicious-model",
            hf_target="target/repo",
            model_class="AutoModel",
            output_attr="last_hidden_state",
            trust_remote_code=True,
        )


def test_onnx_convert_trusted_org_no_namespace():
    # If a trusted org is used without namespace (not typical but possible for some registries)
    # We want to see how it handles it.
    # Current logic: org = hf_source.split("/")[0] if "/" in hf_source else hf_source
    # If "Qwen" is trusted and hf_source="Qwen", org="Qwen" -> OK.
    # If "attacker" is not trusted and hf_source="attacker", org="attacker" -> Fail.
    with pytest.raises(ValueError, match="Untrusted organization"):
        onnx_convert_model(
            model_name="test",
            hf_source="attacker",
            hf_target="target/repo",
            model_class="AutoModel",
            output_attr="last_hidden_state",
            trust_remote_code=True,
        )
