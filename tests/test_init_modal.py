import importlib
from unittest.mock import patch

import ai_workers


def test_init_modal_exception():
    """Verify that an exception in modal.is_local() during package init is suppressed."""
    with patch("modal.is_local", side_effect=Exception("Modal initialization failed")):
        # Reload the package to trigger the initialization logic
        importlib.reload(ai_workers)

    # If we reached here, the exception was suppressed
    assert ai_workers.__version__ is not None


def test_version_metadata_exception():
    """Verify that an exception in version retrieval is suppressed."""
    with patch("importlib.metadata.version", side_effect=Exception("Metadata not found")):
        importlib.reload(ai_workers)

    # Fallback version should be set
    assert ai_workers.__version__ == "0.0.0-dev"


def test_init_modal_success():
    """Verify successful Modal initialization and version retrieval."""
    with (
        patch("modal.is_local", return_value=True),
        patch("importlib.metadata.version", return_value="1.2.3"),
    ):
        importlib.reload(ai_workers)

    assert ai_workers.__version__ == "1.2.3"
