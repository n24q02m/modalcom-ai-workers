"""Coverage boost for uncovered branches in ai_workers (common + workers)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

# ===========================================================================
# utils.py -- is_safe_url branches
# ===========================================================================


class TestUtilsUncoveredBranches:
    """Cover lines 38-40, 67-69."""

    def test_is_safe_url_parse_exception(self):
        """Line 38-40: urlparse raises an exception."""
        from ai_workers.common.utils import is_safe_url

        with patch("ai_workers.common.utils.urlparse", side_effect=ValueError("bad")):
            assert is_safe_url("http://example.com") is None

    def test_is_safe_url_invalid_ip_from_dns(self):
        """Line 67-69: ip_address() raises ValueError for malformed IP."""
        import socket

        from ai_workers.common.utils import is_safe_url

        bad_addrinfo = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("not-an-ip", 0))]
        with patch("socket.getaddrinfo", return_value=bad_addrinfo):
            assert is_safe_url("http://example.com/img.png") is None


# ===========================================================================
# vl_embedding.py -- _last_token_pool + compute methods + serve() branches
# Note: Pydantic models are now module-level to avoid class-not-defined errors in Modal
# ===========================================================================


class TestVLEmbeddingComputeMethods:
    """Cover lines 99-106, 112-145, 157, 178-192, 217-221."""

    def test_last_token_pool_left_padding(self):
        """Lines 99-103: left padding branch."""
        from ai_workers.workers.vl_embedding import VLEmbeddingServer

        attention_mask = MagicMock()
        last_col = MagicMock()
        last_col.sum.return_value = 2  # equals batch_size
        attention_mask.__getitem__ = MagicMock(return_value=last_col)
        attention_mask.shape = (2, 5)

        hidden = MagicMock()
        last_hidden = MagicMock()
        hidden.__getitem__ = MagicMock(return_value=last_hidden)

        result = VLEmbeddingServer._last_token_pool(hidden, attention_mask)
        assert result is last_hidden

    def test_last_token_pool_right_padding(self):
        """Lines 104-108: right padding branch."""
        from ai_workers.workers.vl_embedding import VLEmbeddingServer

        attention_mask = MagicMock()
        last_col = MagicMock()
        last_col.sum.return_value = 1  # != batch_size=2
        attention_mask.__getitem__ = MagicMock(return_value=last_col)
        attention_mask.shape = (2, 5)
        attention_mask.sum.return_value = MagicMock(__sub__=lambda self, x: MagicMock())

        hidden = MagicMock()
        hidden.shape = (2, 5, 8)
        indexed = MagicMock()
        hidden.__getitem__ = MagicMock(return_value=indexed)

        result = VLEmbeddingServer._last_token_pool(hidden, attention_mask)
        assert result is indexed

    def test_embed_text_method(self):
        """Lines 112-145: _embed_text called through endpoint."""
        from ai_workers.workers.vl_embedding import VLEmbeddingServer

        server = VLEmbeddingServer()
        server._embed_text = MagicMock(return_value=[[0.1, 0.2, 0.3]])

        with patch.dict(os.environ, {"API_KEY": "k"}):
            app = server.serve()

        from fastapi.testclient import TestClient

        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/embeddings",
            json={"model": "qwen3-vl-embedding-2b", "input": "hello"},
            headers={"Authorization": "Bearer k"},
        )

        assert resp.status_code == 200
        server._embed_text.assert_called_once()

    def test_embed_multimodal_ssrf_blocked(self):
        """Line 157: _embed_multimodal blocks unsafe URLs."""
        from ai_workers.workers.vl_embedding import VLEmbeddingServer

        server = VLEmbeddingServer()
        server.models = {"qwen3-vl-embedding-2b": MagicMock()}
        server.processors = {"qwen3-vl-embedding-2b": MagicMock()}

        with (
            patch.dict(
                "sys.modules",
                {"qwen_vl_utils": MagicMock()},
            ),
            patch("ai_workers.common.utils.is_safe_url", return_value=None),
            pytest.raises(ValueError, match="SSRF"),
        ):
            server._embed_multimodal(
                "qwen3-vl-embedding-2b", "text", "http://internal.local/img.png"
            )

    def test_embed_multimodal_base64_skips_ssrf(self):
        """Line 156: data: URI skips SSRF check."""
        from ai_workers.workers.vl_embedding import VLEmbeddingServer

        server = VLEmbeddingServer()

        mock_processor = MagicMock()
        mock_processor.apply_chat_template.return_value = "text"

        # Mock the processor call result with attention_mask
        mock_attention_mask = MagicMock()
        mock_attention_mask.shape = (1, 5)
        # Left padding: last col sum == batch_size
        last_col = MagicMock()
        last_col.sum.return_value = 1
        mock_attention_mask.__getitem__ = MagicMock(return_value=last_col)

        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_inputs.__getitem__ = lambda self, key: mock_attention_mask
        mock_processor.return_value = mock_inputs

        # Model output
        mock_outputs = MagicMock()
        mock_model = MagicMock()
        mock_model.return_value = mock_outputs

        server.models = {"qwen3-vl-embedding-2b": mock_model}
        server.processors = {"qwen3-vl-embedding-2b": mock_processor}

        mock_process_vision = MagicMock(return_value=([MagicMock()], None))

        with (
            patch.dict(
                "sys.modules",
                {"qwen_vl_utils": MagicMock(process_vision_info=mock_process_vision)},
            ),
            patch("ai_workers.common.utils.is_safe_url") as mock_ssrf,
        ):
            server._embed_multimodal(
                "qwen3-vl-embedding-2b", "describe", "data:image/png;base64,iVBOR..."
            )
            # is_safe_url should NOT be called for data: URIs
            mock_ssrf.assert_not_called()

    def test_embed_multimodal_via_endpoint(self):
        """Lines 178-192, 217-221: multimodal input through endpoint."""
        from ai_workers.workers.vl_embedding import VLEmbeddingServer

        server = VLEmbeddingServer()
        server._embed_multimodal = MagicMock(return_value=[0.5, 0.6])

        with patch.dict(os.environ, {"API_KEY": "k"}):
            app = server.serve()

        from fastapi.testclient import TestClient

        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/embeddings",
            json={
                "model": "qwen3-vl-embedding-2b",
                "input": {"text": "describe", "image_url": "https://example.com/img.jpg"},
            },
            headers={"Authorization": "Bearer k"},
        )

        assert resp.status_code == 200
        server._embed_multimodal.assert_called_once()

    def test_list_of_vlinputs_mixed_via_endpoint(self):
        """Lines 287-295: list of VLEmbeddingInput with mixed image/text."""
        from ai_workers.workers.vl_embedding import VLEmbeddingServer

        server = VLEmbeddingServer()
        server._embed_multimodal = MagicMock(return_value=[0.9, 0.8])
        server._embed_text = MagicMock(return_value=[[0.1, 0.2]])

        with patch.dict(os.environ, {"API_KEY": "k"}):
            app = server.serve()

        from fastapi.testclient import TestClient

        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/embeddings",
            json={
                "model": "qwen3-vl-embedding-2b",
                "input": [
                    {"text": "with img", "image_url": "http://example.com/img.jpg"},
                    {"text": "no image"},
                ],
            },
            headers={"Authorization": "Bearer k"},
        )

        assert resp.status_code == 200
        assert len(resp.json()["data"]) == 2


# ===========================================================================
# tts.py -- _synthesize non-list return
# ===========================================================================


class TestTTSSynthesizeEdgeCases:
    """Cover edge case where model returns non-list wavs."""

    def test_synthesize_non_list_wavs(self):
        """Line 126: wavs is not a list (e.g., single numpy array)."""
        from ai_workers.workers.tts import TTSServer

        server = TTSServer()
        mock_model = MagicMock()
        # Simulate non-list return (single array instead of list of arrays)
        single_wav = MagicMock()
        mock_model.generate_custom_voice.return_value = (single_wav, 24000)
        server.models = {"qwen3-tts-0.6b": mock_model}

        wavs, sr = server._synthesize("qwen3-tts-0.6b", "hello")
        assert sr == 24000
        # wavs is the single_wav directly (not indexed from list)
        assert wavs is single_wav


# ===========================================================================
# volumes.py -- download_models
# ===========================================================================


class TestDownloadModels:
    """Cover download_models function (lines 82-105)."""

    def test_download_models_success(self):
        """Lines 82-97: successful download of models."""
        mock_hf_hub = MagicMock()
        mock_hf_hub.snapshot_download.return_value = "/cache/model"

        with patch.dict("sys.modules", {"huggingface_hub": mock_hf_hub}):
            from importlib import reload

            from ai_workers.common import volumes

            reload(volumes)

            mock_vol = MagicMock()
            with patch.object(volumes, "hf_cache_vol", mock_vol):
                result = volumes.download_models()

        assert "OK:" in result
        mock_vol.commit.assert_called_once()

    def test_download_models_failure(self):
        """Lines 98-100: one model fails to download."""
        mock_hf_hub = MagicMock()
        mock_hf_hub.snapshot_download.side_effect = RuntimeError("network error")

        with patch.dict("sys.modules", {"huggingface_hub": mock_hf_hub}):
            from importlib import reload

            from ai_workers.common import volumes

            reload(volumes)

            mock_vol = MagicMock()
            with patch.object(volumes, "hf_cache_vol", mock_vol):
                result = volumes.download_models()

        assert "FAIL:" in result
        assert "network error" in result
        mock_vol.commit.assert_called_once()


# ===========================================================================
# __init__.py -- version fallback
# ===========================================================================


class TestPackageInit:
    """Cover __init__.py line 7-9 (version fallback)."""

    def test_version_fallback_when_metadata_unavailable(self):
        """Lines 7-9: when importlib.metadata.version raises, fallback to 0.0.0-dev."""
        with patch("importlib.metadata.version", side_effect=Exception("not found")):
            import importlib

            import ai_workers

            importlib.reload(ai_workers)
            assert ai_workers.__version__ == "0.0.0-dev"
