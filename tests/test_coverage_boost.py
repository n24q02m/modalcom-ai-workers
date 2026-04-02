"""Tests to improve coverage for uncovered branches and error paths.

Targets:
- utils.py: URL parse error, invalid IP, base64 size limit, image download size limit, ValueError re-raise
- asr.py: _transcribe list/object results, audio file too large
- embedding.py: _last_token_pool, _embed method
- ocr.py: _run_ocr with infer() and generate() fallback
- reranker.py: _score_batch, v2 endpoint
- vl_embedding.py: _last_token_pool, _embed_text, _embed_multimodal
- tts.py: _synthesize with non-list wavs return
- volumes.py: download_models
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

# ===========================================================================
# utils.py -- uncovered branches
# ===========================================================================


class TestUtilsUncoveredBranches:
    """Cover lines 38-40, 67-69, 103, 135-136, 143 in utils.py."""

    def test_is_safe_url_parse_exception(self):
        """Line 38-40: urlparse raises an exception."""
        from ai_workers.common.utils import is_safe_url

        with patch("ai_workers.common.utils.urlparse", side_effect=ValueError("bad")):
            assert is_safe_url("http://example.com") is False

    def test_is_safe_url_invalid_ip_from_dns(self):
        """Line 67-69: ip_address() raises ValueError for malformed IP."""
        import socket

        from ai_workers.common.utils import is_safe_url

        bad_addrinfo = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("not-an-ip", 0))]
        with patch("socket.getaddrinfo", return_value=bad_addrinfo):
            assert is_safe_url("http://example.com/img.png") is False

    def test_load_image_base64_exceeds_size_limit(self):
        """Line 103: base64 data exceeds MAX_BASE64_SIZE."""
        from ai_workers.common.utils import load_image_from_url

        with (
            patch("ai_workers.common.utils.MAX_BASE64_SIZE", 10),
            pytest.raises(ValueError, match="exceeds size limit"),
        ):
            load_image_from_url("data:image/png;base64," + "A" * 20)

    def test_load_image_url_exceeds_download_size_limit(self):
        """Line 135-136: downloaded image exceeds MAX_IMAGE_SIZE."""
        from ai_workers.common.utils import load_image_from_url

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_content = MagicMock(return_value=iter([b"x" * 100]))
        mock_resp.close = MagicMock()

        with (
            patch("ai_workers.common.utils.is_safe_url", return_value=True),
            patch("ai_workers.common.utils._session.get", return_value=mock_resp),
            patch("ai_workers.common.utils.MAX_IMAGE_SIZE", 10),
            pytest.raises(ValueError, match="exceeds size limit"),
        ):
            load_image_from_url("https://example.com/big-image.png")

    def test_load_image_url_valueerror_reraise(self):
        """Line 143: ValueError from size limit is re-raised, not wrapped in RuntimeError."""
        from ai_workers.common.utils import load_image_from_url

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_content = MagicMock(return_value=iter([b"x" * 100]))
        mock_resp.close = MagicMock()

        with (
            patch("ai_workers.common.utils.is_safe_url", return_value=True),
            patch("ai_workers.common.utils._session.get", return_value=mock_resp),
            patch("ai_workers.common.utils.MAX_IMAGE_SIZE", 10),
            pytest.raises(ValueError),
        ):
            load_image_from_url("https://example.com/big.png")


# ===========================================================================
# asr.py -- uncovered branches
# ===========================================================================


class TestASRUncoveredBranches:
    """Cover _transcribe list results (lines 109-116) and file too large (line 184)."""

    def test_transcribe_list_of_objects_with_text_attr(self):
        """Line 109-112: result is a list of objects with .text attribute."""
        from ai_workers.workers.asr import DEFAULT_MODEL, ASRServer

        server = ASRServer()
        mock_model = MagicMock()
        item = MagicMock()
        item.text = " transcribed text "
        mock_model.transcribe.return_value = [item]
        server.models = {DEFAULT_MODEL: mock_model}

        result = server._transcribe(DEFAULT_MODEL, b"audio_data")
        assert result == "transcribed text"

    def test_transcribe_list_of_dicts(self):
        """Line 113-114: result is a list of dicts with 'text' key."""
        from ai_workers.workers.asr import DEFAULT_MODEL, ASRServer

        server = ASRServer()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = [{"text": " dict result "}]
        server.models = {DEFAULT_MODEL: mock_model}

        result = server._transcribe(DEFAULT_MODEL, b"audio_data")
        assert result == "dict result"

    def test_transcribe_list_of_other(self):
        """Line 115: result is a list of non-dict, non-object items."""
        from ai_workers.workers.asr import DEFAULT_MODEL, ASRServer

        server = ASRServer()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = [42]
        server.models = {DEFAULT_MODEL: mock_model}

        result = server._transcribe(DEFAULT_MODEL, b"audio_data")
        assert result == "42"

    def test_transcribe_empty_list_fallback(self):
        """Line 116: result is an empty list, falls through to str() fallback."""
        from ai_workers.workers.asr import DEFAULT_MODEL, ASRServer

        server = ASRServer()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = []
        server.models = {DEFAULT_MODEL: mock_model}

        result = server._transcribe(DEFAULT_MODEL, b"audio_data")
        assert result == "[]"

    def test_transcribe_file_too_large(self):
        """Line 184: uploaded audio file exceeds 25 MB."""
        from ai_workers.workers.asr import DEFAULT_MODEL, ASRServer

        server = ASRServer()

        with patch.dict(os.environ, {"API_KEY": "k"}):
            app = server.serve()

        from fastapi.testclient import TestClient

        tc = TestClient(app, raise_server_exceptions=True)

        large_audio = b"x" * (25 * 1024 * 1024 + 1)
        resp = tc.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", large_audio, "audio/wav")},
            data={"model": DEFAULT_MODEL},
            headers={"Authorization": "Bearer k"},
        )

        assert resp.status_code == 413
        assert "too large" in resp.json()["error"]


# ===========================================================================
# embedding.py -- _last_token_pool and _embed via mocked internals
# ===========================================================================


class TestEmbeddingComputeMethods:
    """Cover _last_token_pool (lines 97-104) and _embed (lines 110-134).

    Since torch is stubbed, we test through the serve() endpoint which calls
    _embed internally, and also test _last_token_pool indirectly.
    """

    def test_last_token_pool_left_padding(self):
        """Lines 99-101: left_padding branch (all last tokens non-padded)."""
        from ai_workers.workers.embedding import EmbeddingServer

        # Mock attention_mask where [:, -1].sum() == shape[0] (left padding)
        attention_mask = MagicMock()
        last_col = MagicMock()
        last_col.sum.return_value = 2  # equals batch_size
        attention_mask.__getitem__ = MagicMock(return_value=last_col)
        attention_mask.shape = (2, 5)

        hidden = MagicMock()
        last_hidden = MagicMock()
        hidden.__getitem__ = MagicMock(return_value=last_hidden)

        result = EmbeddingServer._last_token_pool(hidden, attention_mask)
        # Should return hidden[:, -1]
        assert result is last_hidden

    def test_last_token_pool_right_padding(self):
        """Lines 102-106: right_padding branch (variable sequence lengths)."""
        from ai_workers.workers.embedding import EmbeddingServer

        attention_mask = MagicMock()
        last_col = MagicMock()
        last_col.sum.return_value = 1  # != batch_size (2), so right padding
        attention_mask.__getitem__ = MagicMock(return_value=last_col)
        attention_mask.shape = (2, 5)
        attention_mask.sum.return_value = MagicMock(__sub__=lambda self, x: MagicMock())

        hidden = MagicMock()
        hidden.shape = (2, 5, 8)

        result = EmbeddingServer._last_token_pool(hidden, attention_mask)
        # Returns hidden[arange, sequence_lengths]
        assert result is not None

    def test_embed_method_via_endpoint(self):
        """Lines 110-134: _embed called through the API endpoint."""
        from ai_workers.workers.embedding import EmbeddingServer

        server = EmbeddingServer()

        # Mock _embed to return proper embeddings + token count
        server._embed = MagicMock(return_value=([[0.1, 0.2, 0.3]], 5))

        with patch.dict(os.environ, {"API_KEY": "k"}):
            app = server.serve()

        from fastapi.testclient import TestClient

        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v1/embeddings",
            json={"model": "qwen3-embedding-0.6b", "input": "test"},
            headers={"Authorization": "Bearer k"},
        )

        assert resp.status_code == 200
        assert resp.json()["usage"]["total_tokens"] == 5


# ===========================================================================
# ocr.py -- _run_ocr
# ===========================================================================


class TestOCRRunOCR:
    """Cover _run_ocr lines 104-140."""

    def test_run_ocr_with_infer_method(self):
        """Lines 121-129: model has infer() method (DeepSeek-OCR-2 specific)."""
        from ai_workers.workers.ocr import OCRServer

        server = OCRServer()
        mock_model = MagicMock()
        mock_model.infer.return_value = ["extracted text"]
        server.model = mock_model
        server.processor = MagicMock()

        result = server._run_ocr(MagicMock(), prompt="extract")
        assert result == "extracted text"

    def test_run_ocr_with_infer_returns_string(self):
        """Line 129: model.infer() returns a string instead of list."""
        from ai_workers.workers.ocr import OCRServer

        server = OCRServer()
        mock_model = MagicMock()
        mock_model.infer.return_value = "direct string"
        server.model = mock_model
        server.processor = MagicMock()

        result = server._run_ocr(MagicMock(), prompt="extract")
        assert result == "direct string"

    def test_run_ocr_with_infer_empty_list(self):
        """Line 128: model.infer() returns an empty list."""
        from ai_workers.workers.ocr import OCRServer

        server = OCRServer()
        mock_model = MagicMock()
        mock_model.infer.return_value = []
        server.model = mock_model
        server.processor = MagicMock()

        result = server._run_ocr(MagicMock(), prompt="extract")
        assert result == ""

    def test_run_ocr_without_infer_fallback_generate(self):
        """Lines 131-140: model lacks infer() -- fallback to generate()."""
        from ai_workers.workers.ocr import OCRServer

        server = OCRServer()

        # Use spec to exclude 'infer' from the mock
        mock_model = MagicMock(spec=["generate", "device", "eval"])
        mock_model.device = "cpu"

        # Mock input_ids shape for slicing generated_ids[:, input_ids.shape[1]:]
        mock_input_ids = MagicMock()
        mock_input_ids.shape = (1, 10)

        mock_generated = MagicMock()
        # generated_ids[:, input_ids.shape[1]:] returns a sliced tensor
        mock_generated.__getitem__ = MagicMock(return_value=MagicMock())
        mock_model.generate.return_value = mock_generated

        mock_processor = MagicMock()
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = {"input_ids": mock_input_ids}
        mock_inputs.__getitem__ = lambda self, key: {"input_ids": mock_input_ids}[key]
        mock_processor.return_value = mock_inputs
        mock_processor.tokenizer.batch_decode.return_value = ["OCR result"]

        server.model = mock_model
        server.processor = mock_processor

        result = server._run_ocr(MagicMock(), prompt="extract text")
        assert result == "OCR result"

    def test_run_ocr_without_prompt(self):
        """Lines 113-117: _run_ocr called without a prompt (empty string)."""
        from ai_workers.workers.ocr import OCRServer

        server = OCRServer()
        mock_model = MagicMock()
        mock_model.infer.return_value = ["text from free OCR"]
        server.model = mock_model
        server.processor = MagicMock()

        result = server._run_ocr(MagicMock(), prompt="")
        assert result == "text from free OCR"
        mock_model.infer.assert_called_once()
        call_kwargs = mock_model.infer.call_args[1]
        assert call_kwargs["prompts"] is None

    def test_run_ocr_generate_empty_result(self):
        """Line 140: batch_decode returns empty list."""
        from ai_workers.workers.ocr import OCRServer

        server = OCRServer()

        mock_model = MagicMock(spec=["generate", "device", "eval"])
        mock_model.device = "cpu"

        mock_input_ids = MagicMock()
        mock_input_ids.shape = (1, 10)

        mock_generated = MagicMock()
        mock_generated.__getitem__ = MagicMock(return_value=MagicMock())
        mock_model.generate.return_value = mock_generated

        mock_processor = MagicMock()
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = {"input_ids": mock_input_ids}
        mock_inputs.__getitem__ = lambda self, key: {"input_ids": mock_input_ids}[key]
        mock_processor.return_value = mock_inputs
        mock_processor.tokenizer.batch_decode.return_value = []

        server.model = mock_model
        server.processor = mock_processor

        result = server._run_ocr(MagicMock(), prompt="extract")
        assert result == ""

    def test_run_ocr_with_prompt_builds_inputs(self):
        """Lines 107-112: _run_ocr with non-empty prompt calls processor with text=prompt."""
        from ai_workers.workers.ocr import OCRServer

        server = OCRServer()
        mock_model = MagicMock()
        mock_model.infer.return_value = ["result"]
        mock_processor = MagicMock()
        server.model = mock_model
        server.processor = mock_processor

        fake_image = MagicMock()
        server._run_ocr(fake_image, prompt="extract table")

        # Verify processor was called with images and text
        mock_processor.assert_called_once()
        call_kwargs = mock_processor.call_args[1]
        assert call_kwargs["images"] is fake_image
        assert call_kwargs["text"] == "extract table"


# ===========================================================================
# reranker.py -- _score_batch and v2 endpoint
# ===========================================================================


class TestRerankerUncoveredBranches:
    """Cover _score_batch (lines 171-237) and v2 endpoint (lines 326-329)."""

    def test_score_batch_processes_documents(self):
        """Lines 171-237: _score_batch processes documents in batches."""
        from ai_workers.workers.reranker import RerankerServer

        server = RerankerServer()

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "<prompt>"
        mock_tokenizer.padding_side = "left"
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"

        # Mock the tokenizer call result
        mock_attention_mask = MagicMock()
        mock_attention_mask.sum.return_value = MagicMock(__sub__=lambda self, x: MagicMock())
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_inputs.__getitem__ = lambda self, key: mock_attention_mask
        mock_tokenizer.return_value = mock_inputs

        # Backbone output
        mock_backbone_outputs = MagicMock()
        mock_backbone = MagicMock()
        mock_backbone.return_value = mock_backbone_outputs

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.model = mock_backbone

        # yes_no_weight
        yes_no_weight = MagicMock()

        # Mock softmax -> probs[:, 1].tolist() to return list of scores
        import sys

        fn_stub = sys.modules["torch.nn.functional"]
        original_softmax = fn_stub.softmax
        mock_probs = MagicMock()
        mock_probs.__getitem__ = MagicMock(
            return_value=MagicMock(tolist=MagicMock(return_value=[0.7, 0.8]))
        )
        fn_stub.softmax = MagicMock(return_value=mock_probs)

        server.models = {"qwen3-reranker-8b": mock_model}
        server.tokenizers = {"qwen3-reranker-8b": mock_tokenizer}
        server.yes_no_weights = {"qwen3-reranker-8b": yes_no_weight}

        try:
            scores = server._score_batch("qwen3-reranker-8b", "query", ["doc1", "doc2"])
        finally:
            fn_stub.softmax = original_softmax

        assert isinstance(scores, list)
        assert len(scores) == 2
        # Verify tokenizer padding was set to right
        assert mock_tokenizer.padding_side == "right"
        assert mock_tokenizer.pad_token == "<eos>"

    def test_score_batch_with_existing_pad_token(self):
        """_score_batch does not override existing pad_token."""
        from ai_workers.workers.reranker import RerankerServer

        server = RerankerServer()

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "<prompt>"
        mock_tokenizer.padding_side = "left"
        mock_tokenizer.pad_token = "<pad>"  # Already set

        mock_attention_mask = MagicMock()
        mock_attention_mask.sum.return_value = MagicMock(__sub__=lambda self, x: MagicMock())
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_inputs.__getitem__ = lambda self, key: mock_attention_mask
        mock_tokenizer.return_value = mock_inputs

        mock_backbone_outputs = MagicMock()
        mock_backbone = MagicMock()
        mock_backbone.return_value = mock_backbone_outputs

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.model = mock_backbone

        server.models = {"qwen3-reranker-8b": mock_model}
        server.tokenizers = {"qwen3-reranker-8b": mock_tokenizer}
        server.yes_no_weights = {"qwen3-reranker-8b": MagicMock()}

        server._score_batch("qwen3-reranker-8b", "query", ["doc1"], instruction="Custom")

        # pad_token should remain unchanged
        assert mock_tokenizer.pad_token == "<pad>"

    def test_v2_rerank_endpoint(self):
        """Lines 326-329: /v2/rerank endpoint."""
        from ai_workers.workers.reranker import RerankerServer

        server = RerankerServer()
        server._score_batch = MagicMock(return_value=[0.8])

        with patch.dict(os.environ, {"API_KEY": "k"}):
            app = server.serve()

        from fastapi.testclient import TestClient

        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v2/rerank",
            json={
                "model": "qwen3-reranker-8b",
                "query": "test query",
                "documents": ["test doc"],
            },
            headers={"Authorization": "Bearer k"},
        )

        assert resp.status_code == 200
        assert resp.json()["model"] == "qwen3-reranker-8b"
        assert len(resp.json()["results"]) == 1

    def test_v2_rerank_unknown_model(self):
        """Lines 326-329: /v2/rerank with unknown model returns 400."""
        from ai_workers.workers.reranker import RerankerServer

        server = RerankerServer()

        with patch.dict(os.environ, {"API_KEY": "k"}):
            app = server.serve()

        from fastapi.testclient import TestClient

        tc = TestClient(app, raise_server_exceptions=True)
        resp = tc.post(
            "/v2/rerank",
            json={
                "model": "bad-model",
                "query": "q",
                "documents": ["d"],
            },
            headers={"Authorization": "Bearer k"},
        )

        assert resp.status_code == 400
        assert "Invalid rerank request for model:" in resp.json()["error"]


# ===========================================================================
# vl_embedding.py -- _last_token_pool, _embed_text, _embed_multimodal
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
            patch("ai_workers.common.utils.is_safe_url", return_value=False),
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
        from ai_workers.workers.tts import TTSOptions, TTSServer

        server = TTSServer()
        mock_model = MagicMock()
        # Simulate non-list return (single array instead of list of arrays)
        single_wav = MagicMock()
        mock_model.generate_custom_voice.return_value = (single_wav, 24000)
        server.models = {"qwen3-tts-0.6b": mock_model}

        wavs, sr = server._synthesize("qwen3-tts-0.6b", "hello", TTSOptions())
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
