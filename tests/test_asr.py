import sys
from unittest.mock import MagicMock, patch
import pytest
import numpy as np

# We need to mock modal before importing the worker module because
# the worker module uses modal decorators at the module level.
@pytest.fixture
def mock_asr_worker_modules():
    """Mock all heavy dependencies for ASR worker."""
    with patch.dict(sys.modules):
        # Mock modal
        mock_modal = MagicMock()
        mock_app = MagicMock()

        # We need the decorator to return the class/function unmodified
        def identity_decorator(*args, **kwargs):
            def wrapper(obj):
                return obj
            return wrapper

        mock_app.cls.side_effect = identity_decorator
        mock_modal.App.return_value = mock_app
        mock_modal.enter.side_effect = identity_decorator
        mock_modal.asgi_app.side_effect = identity_decorator

        sys.modules["modal"] = mock_modal

        # Mock other dependencies
        sys.modules["torch"] = MagicMock()
        sys.modules["transformers"] = MagicMock()
        sys.modules["librosa"] = MagicMock()
        sys.modules["soundfile"] = MagicMock()

        # Mock internal dependencies that might import modal
        sys.modules["ai_workers.common.images"] = MagicMock()
        sys.modules["ai_workers.common.r2"] = MagicMock()

        # Ensure we re-import the module to apply mocks
        if "ai_workers.workers.asr" in sys.modules:
            del sys.modules["ai_workers.workers.asr"]

        import ai_workers.workers.asr
        yield ai_workers.workers.asr

def test_load_audio(mock_asr_worker_modules):
    """Test _load_audio method."""
    asr_module = mock_asr_worker_modules
    server = asr_module.ASRServer()

    # Setup librosa mock
    # Note: We need to get the mock from sys.modules or the imported module
    # because the module under test imported librosa into its namespace
    # but since we patched sys.modules before import, it should be the same mock

    import librosa

    expected_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    expected_sr = 16000

    # Configure the mock to return our expected data
    librosa.load.return_value = (expected_audio, expected_sr)

    # Test input
    file_bytes = b"fake_audio_content"

    result = server._load_audio(file_bytes)

    # Verify librosa.load was called correctly
    args, kwargs = librosa.load.call_args
    # args[0] should be a BytesIO object
    assert args[0].read() == file_bytes # Check if BytesIO contains correct bytes
    assert kwargs["sr"] == 16000
    assert kwargs["mono"] is True

    # Verify result
    assert np.array_equal(result["raw"], expected_audio)
    assert result["sampling_rate"] == expected_sr
