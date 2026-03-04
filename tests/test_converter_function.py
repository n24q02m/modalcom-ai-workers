"""Tests for converter.convert_model function body.

converter.py has a broken import at module level (MODELS_MOUNT_PATH from images).
We work around this by injecting a stub module into sys.modules and calling
the function body directly.

The function is decorated with @convert_app.function(...) which the modal mock
turns into an identity decorator, so convert_model remains directly callable.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Module-level mock for converter (broken import workaround)
# ---------------------------------------------------------------------------

_MODELS_MOUNT_PATH = "/tmp/test_models"


def _get_mock_converter_module(models_mount_path: str = _MODELS_MOUNT_PATH) -> types.ModuleType:
    """Build a minimal stub that replaces ai_workers.workers.converter."""
    stub = types.ModuleType("ai_workers.workers.converter")
    stub.MODELS_MOUNT_PATH = models_mount_path  # type: ignore[attr-defined]
    return stub


def _make_convert_model_fn():
    """Import convert_model by temporarily mocking the broken module-level import."""
    # Build mock for broken images import
    mock_images = MagicMock()
    mock_images.MODELS_MOUNT_PATH = _MODELS_MOUNT_PATH
    mock_images.converter_image.return_value = MagicMock()

    mock_r2 = MagicMock()
    mock_r2.get_modal_cloud_bucket_mount.return_value = MagicMock()

    with patch.dict(
        sys.modules,
        {
            "ai_workers.common.images": mock_images,
            "ai_workers.common.r2": mock_r2,
        },
    ):
        # Remove cached converter module if it's already loaded
        for key in list(sys.modules):
            if "converter" in key and "workers" in key:
                del sys.modules[key]

        import ai_workers.workers.converter as mod

        return mod.convert_model, mod


# ---------------------------------------------------------------------------
# Tests: skip path (already_exists)
# ---------------------------------------------------------------------------


class TestConvertModelSkipPath:
    """convert_model returns 'skipped' when output already exists and force=False."""

    def test_skipped_when_exists_and_not_force(self, tmp_path: Path) -> None:
        """If output_path exists with files, and force=False, return skipped."""
        model_dir = tmp_path / "qwen3-embedding-0.6b"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(b"x")

        convert_model, mod = _make_convert_model_fn()

        with patch.object(mod, "MODELS_MOUNT_PATH", str(tmp_path)):
            result = convert_model(
                model_name="qwen3-embedding-0.6b",
                hf_id="Qwen/Qwen3-Embedding-0.6B",
                precision="fp16",
                model_class="AutoModel",
                task="feature-extraction",
                trust_remote_code=False,
                extra_load_kwargs={},
                force=False,
            )

        assert result["status"] == "skipped"
        assert result["reason"] == "already_exists"
        assert result["model_name"] == "qwen3-embedding-0.6b"
        assert result["files_count"] == 1

    def test_empty_dir_not_skipped(self, tmp_path: Path) -> None:
        """Empty output directory should NOT trigger skip (no files)."""
        model_dir = tmp_path / "empty-model"
        model_dir.mkdir()

        convert_model, mod = _make_convert_model_fn()
        mock_model = MagicMock()
        mock_processor = MagicMock()

        def fake_save_model(path, **kw):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "model.safetensors").write_bytes(b"x" * 100)

        def fake_save_proc(path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "tokenizer.json").write_bytes(b"t" * 50)

        mock_model.save_pretrained.side_effect = fake_save_model
        mock_processor.save_pretrained.side_effect = fake_save_proc

        mock_torch = MagicMock()
        mock_torch.float16 = "float16"
        mock_torch.bfloat16 = "bfloat16"

        tf_stub = MagicMock()
        tf_stub.AutoModel = MagicMock()
        tf_stub.AutoModel.from_pretrained.return_value = mock_model
        tf_stub.AutoModelForCausalLM = MagicMock()
        tf_stub.AutoModelForImageTextToText = MagicMock()
        tf_stub.AutoModelForSpeechSeq2Seq = MagicMock()
        tf_stub.AutoTokenizer = MagicMock()
        tf_stub.AutoTokenizer.from_pretrained.return_value = mock_processor
        tf_stub.AutoProcessor = MagicMock()

        with (
            patch.object(mod, "MODELS_MOUNT_PATH", str(tmp_path)),
            patch.dict(
                sys.modules,
                {
                    "torch": mock_torch,
                    "transformers": tf_stub,
                    "loguru": MagicMock(logger=MagicMock()),
                },
            ),
        ):
            result = convert_model(
                model_name="empty-model",
                hf_id="Org/empty-model",
                precision="fp16",
                model_class="AutoModel",
                task="feature-extraction",
                trust_remote_code=False,
                extra_load_kwargs={},
                force=False,
            )

        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# Tests: success path — various model classes / tasks
# ---------------------------------------------------------------------------


class TestConvertModelSuccessPath:
    """Tests for convert_model success path with various configurations."""

    @pytest.fixture()
    def mocks(self, tmp_path: Path):
        """Common mock setup for success-path tests."""
        convert_model, mod = _make_convert_model_fn()

        mock_model = MagicMock()
        mock_processor = MagicMock()

        def fake_save_model(path, **kw):
            Path(path).mkdir(parents=True, exist_ok=True)
            # Write > 1 MB so total_size_mb rounds to > 0.0
            (Path(path) / "model.safetensors").write_bytes(b"x" * (1024 * 1024 + 1))

        def fake_save_proc(path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "tokenizer.json").write_bytes(b"t" * 1024)

        mock_model.save_pretrained.side_effect = fake_save_model
        mock_processor.save_pretrained.side_effect = fake_save_proc

        mock_torch = MagicMock()
        mock_torch.float16 = "float16"
        mock_torch.bfloat16 = "bfloat16"

        tf_stub = MagicMock()
        tf_stub.AutoModel = MagicMock()
        tf_stub.AutoModel.from_pretrained.return_value = mock_model
        tf_stub.AutoModelForCausalLM = MagicMock()
        tf_stub.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        tf_stub.AutoModelForImageTextToText = MagicMock()
        tf_stub.AutoModelForImageTextToText.from_pretrained.return_value = mock_model
        tf_stub.AutoModelForSpeechSeq2Seq = MagicMock()
        tf_stub.AutoModelForSpeechSeq2Seq.from_pretrained.return_value = mock_model
        tf_stub.AutoTokenizer = MagicMock()
        tf_stub.AutoTokenizer.from_pretrained.return_value = mock_processor
        tf_stub.AutoProcessor = MagicMock()
        tf_stub.AutoProcessor.from_pretrained.return_value = mock_processor

        return {
            "tmp_path": tmp_path,
            "convert_model": convert_model,
            "mod": mod,
            "mock_model": mock_model,
            "mock_processor": mock_processor,
            "mock_torch": mock_torch,
            "tf_stub": tf_stub,
        }

    def _run(
        self,
        mocks,
        model_name,
        model_class,
        task,
        precision="fp16",
        trust_remote_code=False,
        extra_load_kwargs=None,
    ):
        """Helper to invoke convert_model with standard mocks."""
        with (
            patch.object(mocks["mod"], "MODELS_MOUNT_PATH", str(mocks["tmp_path"])),
            patch.dict(
                sys.modules,
                {
                    "torch": mocks["mock_torch"],
                    "transformers": mocks["tf_stub"],
                    "loguru": MagicMock(logger=MagicMock()),
                },
            ),
        ):
            return mocks["convert_model"](
                model_name=model_name,
                hf_id=f"Org/{model_name}",
                precision=precision,
                model_class=model_class,
                task=task,
                trust_remote_code=trust_remote_code,
                extra_load_kwargs=extra_load_kwargs or {},
                force=True,
            )

    def test_automodel_feature_extraction(self, mocks) -> None:
        result = self._run(mocks, "embed-model", "AutoModel", "feature-extraction")
        assert result["status"] == "success"
        assert result["files_count"] >= 1
        assert result["total_size_mb"] >= 0
        assert "output_path" in result

    def test_automodel_bf16(self, mocks) -> None:
        result = self._run(mocks, "embed-bf16", "AutoModel", "feature-extraction", precision="bf16")
        assert result["status"] == "success"

    def test_causal_lm_feature_extraction(self, mocks) -> None:
        result = self._run(mocks, "reranker-model", "AutoModelForCausalLM", "feature-extraction")
        assert result["status"] == "success"

    def test_vl_task_uses_processor(self, mocks) -> None:
        """vl-embedding task should call AutoProcessor, not AutoTokenizer."""
        result = self._run(mocks, "vl-embed", "AutoModel", "vl-embedding")
        assert result["status"] == "success"
        mocks["tf_stub"].AutoProcessor.from_pretrained.assert_called_once()

    def test_vl_reranker_task_uses_processor(self, mocks) -> None:
        result = self._run(mocks, "vl-rerank", "AutoModel", "vl-reranker")
        assert result["status"] == "success"
        mocks["tf_stub"].AutoProcessor.from_pretrained.assert_called_once()

    def test_asr_task_uses_processor(self, mocks) -> None:
        result = self._run(
            mocks, "asr-model", "AutoModelForSpeechSeq2Seq", "automatic-speech-recognition"
        )
        assert result["status"] == "success"
        mocks["tf_stub"].AutoProcessor.from_pretrained.assert_called_once()

    def test_feature_extraction_uses_tokenizer(self, mocks) -> None:
        result = self._run(mocks, "embed-tok", "AutoModel", "feature-extraction")
        assert result["status"] == "success"
        mocks["tf_stub"].AutoTokenizer.from_pretrained.assert_called_once()

    def test_image_text_to_text_model_class(self, mocks) -> None:
        result = self._run(mocks, "ocr-model", "AutoModelForImageTextToText", "vl-embedding")
        assert result["status"] == "success"

    def test_seq2seq_model_class(self, mocks) -> None:
        result = self._run(
            mocks, "asr-model2", "AutoModelForSpeechSeq2Seq", "automatic-speech-recognition"
        )
        assert result["status"] == "success"

    def test_extra_load_kwargs_passed(self, mocks) -> None:
        """Extra load kwargs should be forwarded to from_pretrained."""
        result = self._run(
            mocks,
            "embed-extra",
            "AutoModel",
            "feature-extraction",
            extra_load_kwargs={"revision": "main"},
        )
        assert result["status"] == "success"
        call_kwargs = mocks["tf_stub"].AutoModel.from_pretrained.call_args.kwargs
        assert call_kwargs.get("revision") == "main"

    def test_trust_remote_code_passed(self, mocks) -> None:
        result = self._run(
            mocks,
            "trusted-model",
            "AutoModel",
            "feature-extraction",
            trust_remote_code=True,
        )
        assert result["status"] == "success"
        call_kwargs = mocks["tf_stub"].AutoModel.from_pretrained.call_args.kwargs
        assert call_kwargs.get("trust_remote_code") is True

    def test_result_has_output_path(self, mocks) -> None:
        result = self._run(mocks, "out-path-test", "AutoModel", "feature-extraction")
        assert "output_path" in result
        assert result["output_path"].endswith("out-path-test")

    def test_files_count_matches_saved_files(self, mocks) -> None:
        """files_count should equal the number of files saved."""
        result = self._run(mocks, "count-test", "AutoModel", "feature-extraction")
        assert result["status"] == "success"
        # save_pretrained writes 1 file, save_proc writes 1 file = 2 total
        assert result["files_count"] == 2

    def test_total_size_mb_positive(self, mocks) -> None:
        result = self._run(mocks, "size-test", "AutoModel", "feature-extraction")
        assert result["total_size_mb"] > 0


# ---------------------------------------------------------------------------
# Tests: error path — unknown model_class raises ImportError
# ---------------------------------------------------------------------------


class TestConvertModelErrors:
    """Tests for convert_model error conditions."""

    def _base_mocks(self):
        mock_torch = MagicMock()
        mock_torch.float16 = "float16"
        mock_torch.bfloat16 = "bfloat16"
        tf_stub = MagicMock()
        tf_stub.AutoModel = MagicMock()
        tf_stub.AutoModelForCausalLM = MagicMock()
        tf_stub.AutoModelForSpeechSeq2Seq = MagicMock()
        tf_stub.AutoModelForImageTextToText = MagicMock()
        return mock_torch, tf_stub

    def test_unknown_model_class_raises_import_error(self, tmp_path: Path) -> None:
        """Invalid model_class should raise ImportError."""
        convert_model, mod = _make_convert_model_fn()
        mock_torch, tf_stub = self._base_mocks()

        with (
            patch.object(mod, "MODELS_MOUNT_PATH", str(tmp_path)),
            patch.dict(
                sys.modules,
                {
                    "torch": mock_torch,
                    "transformers": tf_stub,
                    "loguru": MagicMock(logger=MagicMock()),
                },
            ),
            pytest.raises(ImportError, match="Model class"),
        ):
            convert_model(
                model_name="test",
                hf_id="Org/test",
                precision="fp16",
                model_class="UnknownModelClass",
                task="feature-extraction",
                trust_remote_code=False,
                extra_load_kwargs={},
                force=True,
            )

    def test_none_model_class_when_not_available(self, tmp_path: Path) -> None:
        """When AutoModelForImageTextToText is None (old transformers), class maps to None -> ImportError."""
        convert_model, mod = _make_convert_model_fn()
        mock_torch, tf_stub = self._base_mocks()

        # Simulate transformers where AutoModelForImageTextToText import raises ImportError
        # The function does: try: from transformers import AutoModelForImageTextToText
        #                    except ImportError: AutoModelForImageTextToText = None
        # Then model_class_map["AutoModelForImageTextToText"] = None -> cls is None -> ImportError
        tf_stub.AutoModelForImageTextToText = None  # simulates old transformers

        with (
            patch.object(mod, "MODELS_MOUNT_PATH", str(tmp_path)),
            patch.dict(
                sys.modules,
                {
                    "torch": mock_torch,
                    "transformers": tf_stub,
                    "loguru": MagicMock(logger=MagicMock()),
                },
            ),
            pytest.raises(ImportError),
        ):
            convert_model(
                model_name="test",
                hf_id="Org/test",
                precision="fp16",
                model_class="AutoModelForImageTextToText",
                task="vl-embedding",
                trust_remote_code=False,
                extra_load_kwargs={},
                force=True,
            )
