"""Shared pytest fixtures and sys.modules mocks.

Injects mock stubs for heavy deps (modal, torch, transformers, etc.)
into sys.modules BEFORE any worker/image module is imported.
This lets us import and test worker FastAPI apps without GPU or heavy packages.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from unittest.mock import MagicMock

import pytest


def _make_modal_mock() -> MagicMock:
    """Build a rich modal mock that satisfies decorator usage at module level."""
    modal = MagicMock(name="modal")

    # modal.App() returns a mock that acts as a decorator context
    mock_app = MagicMock()
    mock_app.cls = lambda **kw: lambda cls: cls
    mock_app.function = lambda **kw: lambda fn: fn
    modal.App.return_value = mock_app

    # modal.concurrent returns identity decorator
    modal.concurrent.return_value = lambda cls: cls

    # modal.enter / asgi_app / method are identity decorators
    modal.enter.return_value = lambda fn: fn
    modal.asgi_app.return_value = lambda fn: fn
    modal.method.return_value = lambda fn: fn

    # modal.Secret, modal.CloudBucketMount, modal.Image placeholders
    modal.Secret.from_name.return_value = MagicMock()
    modal.CloudBucketMount.return_value = MagicMock()
    modal.Image.debian_slim.return_value = MagicMock()

    # modal.enable_output context manager
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=None)
    cm.__exit__ = MagicMock(return_value=False)
    modal.enable_output.return_value = cm

    # modal.exception.AuthError subclass so except modal.exception.AuthError works
    exc_mod = MagicMock()
    exc_mod.AuthError = type("AuthError", (Exception,), {})
    modal.exception = exc_mod

    return modal


def _inject_modal() -> None:
    """Put modal mock into sys.modules if modal not installed."""
    if "modal" in sys.modules and not isinstance(sys.modules["modal"], MagicMock):
        # real modal installed — patch it to behave correctly for our tests
        return
    modal_mock = _make_modal_mock()
    sys.modules["modal"] = modal_mock
    sys.modules["modal.exception"] = modal_mock.exception


# Inject immediately at import time so all worker modules load cleanly.
_inject_modal()

# ---------------------------------------------------------------------------
# Torch stub (lightweight — avoids 10GB install)
# ---------------------------------------------------------------------------


def _ensure_torch_stub() -> None:
    """Inject a minimal torch stub if torch is not installed."""
    if "torch" in sys.modules and not isinstance(sys.modules["torch"], MagicMock):
        return  # real torch available

    torch_stub = types.ModuleType("torch")
    torch_stub.float16 = "float16"
    torch_stub.bfloat16 = "bfloat16"
    torch_stub.float32 = "float32"
    torch_stub.no_grad = MagicMock(
        return_value=MagicMock(
            __enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False)
        )
    )

    nn_stub = types.ModuleType("torch.nn")
    functional_stub = types.ModuleType("torch.nn.functional")

    # Configure normalize to return a mock supporting [:, :N].cpu().tolist() → [[0.5, 0.5]]
    _indexed_norm = MagicMock()
    _indexed_norm.cpu = MagicMock(
        return_value=MagicMock(tolist=MagicMock(return_value=[[0.5, 0.5]]))
    )
    _norm_result = MagicMock()
    _norm_result.__getitem__ = MagicMock(return_value=_indexed_norm)
    functional_stub.normalize = MagicMock(return_value=_norm_result)

    nn_stub.functional = functional_stub
    nn_stub.Module = object
    torch_stub.nn = nn_stub

    torch_stub.Tensor = MagicMock
    torch_stub.no_grad = MagicMock()
    torch_stub.no_grad.return_value.__enter__ = MagicMock(return_value=None)
    torch_stub.no_grad.return_value.__exit__ = MagicMock(return_value=False)
    torch_stub.sigmoid = MagicMock(return_value=MagicMock(item=MagicMock(return_value=0.9)))

    # Tensor creation functions — return MagicMock that supports tensor-like ops
    torch_stub.ones = MagicMock(return_value=MagicMock())
    torch_stub.tensor = MagicMock(return_value=MagicMock())
    torch_stub.randn = MagicMock(return_value=MagicMock())
    torch_stub.stack = MagicMock(return_value=MagicMock())
    torch_stub.zeros = MagicMock(return_value=MagicMock())

    onnx_stub = types.ModuleType("torch.onnx")
    onnx_stub.export = MagicMock()
    torch_stub.onnx = onnx_stub

    sys.modules["torch"] = torch_stub
    sys.modules["torch.nn"] = nn_stub
    sys.modules["torch.nn.functional"] = functional_stub
    sys.modules["torch.onnx"] = onnx_stub


_ensure_torch_stub()


# ---------------------------------------------------------------------------
# Requests stub (lightweight — avoids installing requests)
# ---------------------------------------------------------------------------


def _ensure_requests_stub() -> None:
    """Inject a minimal requests stub if requests is not installed."""
    if "requests" in sys.modules:
        return

    requests_stub = types.ModuleType("requests")
    requests_stub.get = MagicMock(return_value=MagicMock())
    requests_stub.post = MagicMock(return_value=MagicMock())
    requests_stub.Session = MagicMock()
    sys.modules["requests"] = requests_stub


_ensure_requests_stub()


# ---------------------------------------------------------------------------
# PIL stub (lightweight — avoids installing Pillow)
# ---------------------------------------------------------------------------


def _ensure_pil_stub() -> None:
    """Inject a minimal PIL stub if Pillow is not installed."""
    if "PIL" in sys.modules:
        return
    # Only inject stub if PIL is not actually installed
    if importlib.util.find_spec("PIL") is not None:
        return  # real PIL available, let it be imported naturally
    pil_stub = types.ModuleType("PIL")
    image_stub = types.ModuleType("PIL.Image")
    image_stub.open = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]
    image_stub.new = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]
    pil_stub.Image = image_stub  # type: ignore[attr-defined]
    sys.modules["PIL"] = pil_stub
    sys.modules["PIL.Image"] = image_stub


_ensure_pil_stub()


# ---------------------------------------------------------------------------
# Transformers stub
# ---------------------------------------------------------------------------


def _ensure_transformers_stub() -> None:
    if "transformers" in sys.modules and not isinstance(sys.modules["transformers"], MagicMock):
        return

    tf = types.ModuleType("transformers")
    for cls_name in [
        "AutoModel",
        "AutoTokenizer",
        "AutoProcessor",
        "AutoConfig",
        "AutoModelForCausalLM",
        "AutoModelForImageTextToText",
        "AutoModelForSpeechSeq2Seq",
        "pipeline",
    ]:
        setattr(tf, cls_name, MagicMock(name=cls_name))

    sys.modules["transformers"] = tf


_ensure_transformers_stub()


# ---------------------------------------------------------------------------
# qwen_tts stub
# ---------------------------------------------------------------------------


def _ensure_qwen_tts_stub() -> None:
    """Inject a minimal qwen_tts stub if qwen-tts is not installed."""
    if "qwen_tts" in sys.modules and not isinstance(sys.modules["qwen_tts"], MagicMock):
        return
    if importlib.util.find_spec("qwen_tts") is not None:
        return

    qwen_tts_stub = types.ModuleType("qwen_tts")
    qwen_tts_stub.Qwen3TTSModel = MagicMock(name="Qwen3TTSModel")
    sys.modules["qwen_tts"] = qwen_tts_stub


_ensure_qwen_tts_stub()


# ---------------------------------------------------------------------------
# qwen_asr stub
# ---------------------------------------------------------------------------


def _ensure_qwen_asr_stub() -> None:
    """Inject a minimal qwen_asr stub if qwen-asr is not installed."""
    if "qwen_asr" in sys.modules and not isinstance(sys.modules["qwen_asr"], MagicMock):
        return
    if importlib.util.find_spec("qwen_asr") is not None:
        return

    qwen_asr_stub = types.ModuleType("qwen_asr")
    qwen_asr_stub.Qwen3ASRModel = MagicMock(name="Qwen3ASRModel")
    sys.modules["qwen_asr"] = qwen_asr_stub


_ensure_qwen_asr_stub()


# ---------------------------------------------------------------------------
# soundfile stub
# ---------------------------------------------------------------------------


def _ensure_soundfile_stub() -> None:
    """Inject a minimal soundfile stub if soundfile is not installed."""
    if "soundfile" in sys.modules:
        return
    if importlib.util.find_spec("soundfile") is not None:
        return

    sf_stub = types.ModuleType("soundfile")
    sf_stub.write = MagicMock()
    sf_stub.read = MagicMock(return_value=(MagicMock(), 24000))
    sys.modules["soundfile"] = sf_stub


_ensure_soundfile_stub()


# ---------------------------------------------------------------------------
# numpy stub
# ---------------------------------------------------------------------------


def _ensure_numpy_stub() -> None:
    """Inject a minimal numpy stub if numpy is not installed."""
    if "numpy" in sys.modules:
        return
    if importlib.util.find_spec("numpy") is not None:
        return

    np_stub = types.ModuleType("numpy")
    np_stub.ndarray = type("ndarray", (), {})
    np_stub.zeros = MagicMock(return_value=MagicMock())
    np_stub.array = MagicMock(return_value=MagicMock())
    np_stub.float32 = "float32"
    np_stub.isscalar = lambda obj: isinstance(obj, (int, float, complex, str, bytes))
    np_stub.bool_ = type("bool_", (int,), {})
    sys.modules["numpy"] = np_stub


_ensure_numpy_stub()


# ---------------------------------------------------------------------------
# Autouse fixture: ensure WORKER_API_KEY is set so auth is enforced by default.
# Tests that explicitly need dev-mode (no key) override this via patch.dict.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _default_worker_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set a sentinel WORKER_API_KEY for every test so auth is enforced.

    Tests that want dev-mode (skip auth) explicitly patch the env vars to empty
    or unset via ``unittest.mock.patch.dict``, which overrides this fixture.
    """
    if not os.environ.get("API_KEY") and not os.environ.get("WORKER_API_KEY"):
        monkeypatch.setenv("WORKER_API_KEY", "k")


@pytest.fixture(autouse=True)
def _reset_auth_cache():
    import ai_workers.common.auth

    ai_workers.common.auth._valid_keys = None
