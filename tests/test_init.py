import importlib
from unittest.mock import MagicMock, patch

import ai_workers


def test_init_modal_exception_suppressed():
    """Verify that exceptions during modal initialization are suppressed."""
    with patch.dict("sys.modules", {"modal": MagicMock()}):
        import modal

        modal.is_local.side_effect = Exception("Modal init error")

        # Reload to trigger the init code
        importlib.reload(ai_workers)

        # Should still be importable and have version
        assert hasattr(ai_workers, "__version__")
        modal.is_local.assert_called()


def test_init_modal_import_error_suppressed():
    """Verify that ImportError for modal is suppressed."""
    with patch.dict("sys.modules", {"modal": None}):
        importlib.reload(ai_workers)
        assert hasattr(ai_workers, "__version__")


def test_init_version_exception_fallback():
    """Verify fallback to 0.0.0-dev when version retrieval fails."""
    # Ensure __version__ is not already in globals for this test
    if hasattr(ai_workers, "__version__"):
        del ai_workers.__version__

    with patch("importlib.metadata.version", side_effect=Exception("Metadata error")):
        importlib.reload(ai_workers)
        assert ai_workers.__version__ == "0.0.0-dev"


def test_init_version_success():
    """Verify __version__ is set correctly when metadata is available."""
    # Ensure __version__ is not already in globals for this test
    if hasattr(ai_workers, "__version__"):
        del ai_workers.__version__

    with patch("importlib.metadata.version", return_value="1.2.3"):
        importlib.reload(ai_workers)
        assert ai_workers.__version__ == "1.2.3"
