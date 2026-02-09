"""Tests for OCR worker."""

import sys
from unittest.mock import MagicMock

# Mock modal before importing anything from src
mock_modal = MagicMock()
sys.modules["modal"] = mock_modal

# Mock loguru
mock_loguru = MagicMock()
sys.modules["loguru"] = mock_loguru


# Mock decorators to return the class/function unmodified
def cls_decorator(*args, **kwargs):
    def wrapper(cls):
        return cls

    return wrapper


mock_modal.App.return_value.cls.side_effect = cls_decorator
mock_modal.asgi_app.return_value = lambda f: f
mock_modal.enter.return_value = lambda f: f

# Now import the worker
from ai_workers.workers.ocr import OCRServer  # noqa: E402


class TestOCRProcessImageContent:
    """Test _process_image_content method of OCRServer."""

    def test_single_text_part(self):
        """Test extracting text from a single text part."""
        content = [{"type": "text", "text": "Hello world"}]
        server = OCRServer()
        text, url = server._process_image_content(content)
        assert text == "Hello world"
        assert url is None

    def test_single_image_part(self):
        """Test extracting image URL from a single image part."""
        content = [{"type": "image_url", "image_url": {"url": "http://example.com/image.png"}}]
        server = OCRServer()
        text, url = server._process_image_content(content)
        assert text == ""
        assert url == "http://example.com/image.png"

    def test_text_and_image(self):
        """Test extracting both text and image URL."""
        content = [
            {"type": "text", "text": "Analyze this"},
            {"type": "image_url", "image_url": {"url": "http://example.com/image.png"}},
        ]
        server = OCRServer()
        text, url = server._process_image_content(content)
        assert text == "Analyze this"
        assert url == "http://example.com/image.png"

    def test_empty_text(self):
        """Test handling of empty text content."""
        content = [{"type": "text", "text": ""}]
        server = OCRServer()
        text, url = server._process_image_content(content)
        assert text == ""
        assert url is None

    def test_multiple_text_parts_last_wins(self):
        """Verify that the last text part overwrites previous ones."""
        content = [
            {"type": "text", "text": "First"},
            {"type": "text", "text": "Second"},
        ]
        server = OCRServer()
        text, url = server._process_image_content(content)
        assert text == "Second"
        assert url is None

    def test_multiple_images_last_wins(self):
        """Verify that the last image part overwrites previous ones."""
        content = [
            {"type": "image_url", "image_url": {"url": "url1"}},
            {"type": "image_url", "image_url": {"url": "url2"}},
        ]
        server = OCRServer()
        text, url = server._process_image_content(content)
        assert text == ""
        assert url == "url2"

    def test_unknown_type_ignored(self):
        """Test that unknown content types are ignored."""
        content = [{"type": "unknown", "text": "ignored"}]
        server = OCRServer()
        text, url = server._process_image_content(content)
        assert text == ""
        assert url is None

    def test_missing_fields(self):
        """Test handling of missing fields in content parts."""
        content = [
            {"type": "text"},  # Missing "text", should default to ""
            {
                "type": "image_url"
            },  # Missing "image_url" dict, should default to empty dict -> empty url
        ]
        server = OCRServer()
        text, url = server._process_image_content(content)
        assert text == ""
        assert url == ""  # Defaults to empty string
