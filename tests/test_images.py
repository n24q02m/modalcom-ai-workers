"""Tests for common/images.py — Modal image builders."""


def test_transformers_image_default():
    """transformers_image() with flash_attn=False returns a mock modal.Image chain."""
    from ai_workers.common.images import transformers_image

    img = transformers_image()
    # Modal is mocked — result should be a MagicMock
    assert img is not None


def test_transformers_image_flash_attn():
    """transformers_image(flash_attn=True) exercises the flash-attn branch."""
    from ai_workers.common.images import transformers_image

    img = transformers_image(flash_attn=True)
    assert img is not None


def test_transformers_tts_image():
    from ai_workers.common.images import transformers_tts_image

    img = transformers_tts_image()
    assert img is not None


def test_transformers_asr_image():
    from ai_workers.common.images import transformers_asr_image

    img = transformers_asr_image()
    assert img is not None


def test_onnx_converter_image():
    from ai_workers.common.images import onnx_converter_image

    img = onnx_converter_image()
    assert img is not None


def test_gguf_converter_image():
    from ai_workers.common.images import gguf_converter_image

    img = gguf_converter_image()
    assert img is not None


def test_python_version_constant():
    from ai_workers.common.images import PYTHON_VERSION

    assert PYTHON_VERSION == "3.13"
