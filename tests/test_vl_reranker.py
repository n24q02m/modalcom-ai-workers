import sys
from unittest.mock import MagicMock

# --- Mocking infrastructure ---
mock_modal = MagicMock()
sys.modules["modal"] = mock_modal

# Mock modal.App and its decorators
def identity(x, **kwargs):
    return x

mock_app_instance = MagicMock()
mock_app_instance.cls.return_value = identity
mock_modal.App.return_value = mock_app_instance
mock_modal.enter.return_value = identity
mock_modal.asgi_app.return_value = identity
mock_modal.Secret.from_name.return_value = MagicMock()

# Mock modal.Image
mock_image = MagicMock()
mock_modal.Image.debian_slim.return_value = mock_image
mock_image.pip_install.return_value = mock_image
mock_image.env.return_value = mock_image
mock_image.apt_install.return_value = mock_image

# Mock CloudBucketMount
mock_modal.CloudBucketMount = MagicMock()

# Mock torch if not present
try:
    import torch
except ImportError:
    sys.modules["torch"] = MagicMock()

# Mock transformers if not present
try:
    import transformers
except ImportError:
    sys.modules["transformers"] = MagicMock()

# Mock fastapi if not present
try:
    import fastapi
except ImportError:
    sys.modules["fastapi"] = MagicMock()
    mock_fastapi = sys.modules["fastapi"]
    mock_fastapi.FastAPI.return_value = MagicMock()
    mock_fastapi.Request = MagicMock()

# Mock pydantic if not present
try:
    import pydantic
except ImportError:
    sys.modules["pydantic"] = MagicMock()
    # Mock BaseModel to be subclassable
    class MockBaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    sys.modules["pydantic"].BaseModel = MockBaseModel

# Mock loguru if not present
try:
    import loguru
except ImportError:
    sys.modules["loguru"] = MagicMock()

# --- End mocking ---

from ai_workers.workers.vl_reranker import (
    VLRerankerBase,
    VLRerankerLightServer,
    VLRerankerHeavyServer,
    RerankRequest,
    RerankResponse
)

def test_inheritance():
    assert issubclass(VLRerankerLightServer, VLRerankerBase)
    assert issubclass(VLRerankerHeavyServer, VLRerankerBase)

def test_model_names():
    assert VLRerankerLightServer.model_name == "qwen3-vl-reranker-2b"
    assert VLRerankerHeavyServer.model_name == "qwen3-vl-reranker-8b"
    assert VLRerankerLightServer.display_name == "Light"
    assert VLRerankerHeavyServer.display_name == "Heavy"

def test_pydantic_models():
    # Verify models are available and have expected fields
    req = RerankRequest(query="q", documents=["d"], model="m")
    assert req.query == "q"
    assert req.documents == ["d"]
    assert req.model == "m"

def test_base_methods():
    # Test that base class has the expected methods
    assert hasattr(VLRerankerBase, "load_model")
    assert hasattr(VLRerankerBase, "_score_pair")
    assert hasattr(VLRerankerBase, "serve")
    assert hasattr(VLRerankerBase, "create_app")
