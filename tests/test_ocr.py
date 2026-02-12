import base64
import binascii
import io
import sys
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

# --- Mock Modal Setup ---
mock_modal = MagicMock()


def identity(x):
    return x


mock_modal.App.return_value.cls.return_value = identity
mock_modal.App.return_value.asgi_app.return_value = identity
mock_modal.enter.return_value = identity
mock_modal.asgi_app.return_value = identity
mock_modal.Secret.from_name.return_value = MagicMock()

# --- Mock PIL Setup ---
mock_pil = MagicMock()
mock_pil_image_module = MagicMock()
mock_pil.Image = mock_pil_image_module

mock_open = MagicMock()
mock_pil_image_module.open = mock_open

mock_opened_image = MagicMock()
mock_converted_image = MagicMock()
mock_opened_image.convert.return_value = mock_converted_image
mock_open.return_value = mock_opened_image


class UnidentifiedImageError(Exception):
    pass


mock_pil_image_module.UnidentifiedImageError = UnidentifiedImageError

# Apply mocks to sys.modules for the initial import
with patch.dict(
    sys.modules, {"modal": mock_modal, "PIL": mock_pil, "PIL.Image": mock_pil_image_module}
):
    from ai_workers.workers.ocr import OCRServer


class TestOCRServer(unittest.TestCase):
    def setUp(self):
        self.server = OCRServer()

        # Patch sys.modules so 'from PIL import Image' works inside methods
        self.modules_patcher = patch.dict(
            sys.modules, {"PIL": mock_pil, "PIL.Image": mock_pil_image_module}
        )
        self.modules_patcher.start()

        # Reset mocks
        mock_open.reset_mock()
        mock_open.side_effect = None
        mock_opened_image.reset_mock()
        mock_converted_image.reset_mock()

    def tearDown(self):
        self.modules_patcher.stop()

    def create_test_image_bytes(self):
        """Helper to create dummy image bytes."""
        return b"fake_image_bytes"

    def test_load_image_from_base64_valid(self):
        """Test loading a valid base64 encoded image."""
        image_bytes = self.create_test_image_bytes()
        b64_data = base64.b64encode(image_bytes).decode("utf-8")
        url = f"data:image/png;base64,{b64_data}"

        image = self.server._load_image_from_url(url)
        # Verify Image.open was called with BytesIO wrapping our bytes
        mock_open.assert_called()
        args, _ = mock_open.call_args
        self.assertIsInstance(args[0], io.BytesIO)
        self.assertEqual(args[0].getvalue(), image_bytes)

        # Verify .convert("RGB") was called
        mock_opened_image.convert.assert_called_with("RGB")

        # Verify return value
        self.assertEqual(image, mock_converted_image)

    def test_load_image_from_base64_invalid_data(self):
        """Test loading invalid base64 data."""
        url = "data:image/png;base64,invalid_base64_data"

        with self.assertRaises(binascii.Error):
            self.server._load_image_from_url(url)

    @patch("urllib.request.urlopen")
    def test_load_image_from_url_valid(self, mock_urlopen):
        """Test loading a valid image from a URL."""
        image_bytes = self.create_test_image_bytes()

        mock_response = MagicMock()
        mock_response.read.return_value = image_bytes
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None

        mock_urlopen.return_value = mock_response

        url = "http://example.com/image.png"
        image = self.server._load_image_from_url(url)
        mock_urlopen.assert_called_once_with(url)

        mock_open.assert_called()
        args, _ = mock_open.call_args
        self.assertEqual(args[0].getvalue(), image_bytes)

        self.assertEqual(image, mock_converted_image)

    @patch("urllib.request.urlopen")
    def test_load_image_from_url_error(self, mock_urlopen):
        """Test handling of URL errors."""
        mock_urlopen.side_effect = urllib.error.URLError("Not Found")

        with self.assertRaises(urllib.error.URLError):
            self.server._load_image_from_url("http://example.com/missing.png")

    @patch("urllib.request.urlopen")
    def test_load_image_from_url_invalid_content(self, mock_urlopen):
        """Test loading non-image content from URL."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"<html>Not an image</html>"
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None

        mock_urlopen.return_value = mock_response

        # Configure mock_open to raise UnidentifiedImageError
        mock_open.side_effect = UnidentifiedImageError("Cannot identify image file")

        with self.assertRaises(UnidentifiedImageError):
            self.server._load_image_from_url("http://example.com/not_image.html")
