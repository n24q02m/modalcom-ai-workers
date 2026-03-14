from ai_workers.common.config import ModelConfig, Task, Tier


def test_trust_remote_code_default_secure():
    """Verify that trust_remote_code defaults to False for new configurations."""
    config = ModelConfig(
        name="test-model",
        hf_id="test/model",
        task=Task.EMBEDDING,
        tier=Tier.LIGHT,
    )
    # This should fail before the fix, pass after
    assert config.trust_remote_code is False


def test_qwen_models_trusted():
    """Verify that Qwen3 models are explicitly trusted (legacy behavior)."""
    from ai_workers.common.config import list_models

    models = list_models()
    qwen_models = [m for m in models if "qwen3" in m.name]
    assert len(qwen_models) > 0
    for m in qwen_models:
        assert m.trust_remote_code is True, f"{m.name} should trust remote code"


def test_whisper_untrusted():
    """Verify that Whisper is untrusted."""
    from ai_workers.common.config import get_model

    m = get_model("whisper-large-v3")
    assert m.trust_remote_code is False


def test_ocr_model_trusted():
    """Verify that DeepSeek OCR model is explicitly trusted."""
    from ai_workers.common.config import get_model

    m = get_model("deepseek-ocr-2")
    assert m.trust_remote_code is True, "deepseek-ocr-2 should trust remote code"
