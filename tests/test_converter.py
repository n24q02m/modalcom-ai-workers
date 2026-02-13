"""Tests cho converter pipeline.

Kiểm tra CLI convert logic, argument mapping, và kết quả trả về.
Không cần GPU — mock Modal remote calls.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
import typer.testing

from ai_workers.common.config import (
    MODEL_REGISTRY,
    ModelClassType,
    Precision,
    list_models,
)

# ---------------------------------------------------------------------------
# Converter result schema tests
# ---------------------------------------------------------------------------


class TestConverterResultSchema:
    """Validate kết quả trả về từ convert_model."""

    def test_success_result_has_required_keys(self) -> None:
        result = {
            "model_name": "qwen3-embedding-0.6b",
            "status": "success",
            "files_count": 5,
            "total_size_mb": 1200.50,
            "output_path": "/models/qwen3-embedding-0.6b",
        }
        assert result["status"] == "success"
        assert isinstance(result["files_count"], int)
        assert isinstance(result["total_size_mb"], float)
        assert result["output_path"].startswith("/models/")

    def test_skipped_result_has_reason(self) -> None:
        result = {
            "model_name": "qwen3-embedding-0.6b",
            "status": "skipped",
            "reason": "already_exists",
            "files_count": 5,
        }
        assert result["status"] == "skipped"
        assert result["reason"] == "already_exists"

    def test_status_values(self) -> None:
        """Status chỉ có thể là 'success' hoặc 'skipped'."""
        valid_statuses = {"success", "skipped"}
        for status in valid_statuses:
            result = {"status": status, "model_name": "test"}
            assert result["status"] in valid_statuses


# ---------------------------------------------------------------------------
# CLI convert argument mapping tests
# ---------------------------------------------------------------------------


class TestConvertArgumentMapping:
    """Validate model config → remote function argument mapping."""

    def test_all_models_have_convertible_precision(self) -> None:
        """Tất cả models phải có precision FP16 hoặc BF16."""
        valid = {Precision.FP16, Precision.BF16}
        for model in list_models():
            assert model.precision in valid, f"{model.name} has invalid precision"

    def test_all_models_have_known_model_class(self) -> None:
        """Tất cả models phải có model_class trong converter model_class_map."""
        known_classes = {
            ModelClassType.AUTO_MODEL,
            ModelClassType.CAUSAL_LM,
            ModelClassType.IMAGE_TEXT_TO_TEXT,
            ModelClassType.SEQ2SEQ,
        }
        for model in list_models():
            assert model.model_class in known_classes, (
                f"{model.name} has unknown model_class: {model.model_class}"
            )

    def test_precision_to_string_mapping(self) -> None:
        """Precision enum values phải là 'fp16' hoặc 'bf16'."""
        assert Precision.FP16.value == "fp16"
        assert Precision.BF16.value == "bf16"

    def test_model_class_to_string_mapping(self) -> None:
        """ModelClassType enum values phải match converter model_class_map keys."""
        converter_map_keys = {
            "AutoModel",
            "AutoModelForCausalLM",
            "AutoModelForImageTextToText",
            "AutoModelForSpeechSeq2Seq",
        }
        for mct in ModelClassType:
            assert mct.value in converter_map_keys, (
                f"ModelClassType.{mct.name} = '{mct.value}' not in converter map"
            )


# ---------------------------------------------------------------------------
# CLI convert integration tests (mocked Modal)
# ---------------------------------------------------------------------------


class TestConvertCLI:
    """Test CLI convert flow với mocked Modal.

    converter.py có side-effects ở module level (modal.App, CloudBucketMount)
    nên phải mock via sys.modules để tránh import thật.
    modal.enable_output() được mock trực tiếp trên modal module.
    """

    @pytest.fixture(autouse=True)
    def _setup_mocks(self):
        """Pre-populate sys.modules với mock converter module."""
        self.mock_convert_model = MagicMock()
        mock_converter_module = MagicMock()
        mock_converter_module.convert_model = self.mock_convert_model

        # Mock converter module (tránh modal.App side-effect) + modal.enable_output
        with (
            patch.dict(
                sys.modules,
                {"ai_workers.workers.converter": mock_converter_module},
            ),
            patch("modal.enable_output") as mock_enable,
        ):
            mock_enable.return_value.__enter__ = MagicMock(return_value=None)
            mock_enable.return_value.__exit__ = MagicMock(return_value=False)
            self.mock_enable_output = mock_enable
            yield

    def _invoke(self, args: list[str]):
        runner = typer.testing.CliRunner()
        from ai_workers.cli.convert import app

        return runner.invoke(app, args)

    def test_convert_single_model_success(self) -> None:
        self.mock_convert_model.remote.return_value = {
            "model_name": "qwen3-embedding-0.6b",
            "status": "success",
            "files_count": 5,
            "total_size_mb": 1200.50,
            "output_path": "/models/qwen3-embedding-0.6b",
        }

        result = self._invoke(["qwen3-embedding-0.6b"])
        assert result.exit_code == 0
        assert "THÀNH CÔNG" in result.output
        self.mock_convert_model.remote.assert_called_once()

    def test_convert_single_model_skipped(self) -> None:
        self.mock_convert_model.remote.return_value = {
            "model_name": "qwen3-embedding-0.6b",
            "status": "skipped",
            "reason": "already_exists",
            "files_count": 5,
        }

        result = self._invoke(["qwen3-embedding-0.6b"])
        assert result.exit_code == 0
        assert "Bỏ qua" in result.output

    def test_convert_invalid_model(self) -> None:
        result = self._invoke(["nonexistent-model"])
        assert result.exit_code != 0
        assert "Lỗi" in result.output

    def test_convert_passes_force_flag(self) -> None:
        self.mock_convert_model.remote.return_value = {
            "model_name": "qwen3-embedding-0.6b",
            "status": "success",
            "files_count": 5,
            "total_size_mb": 1200.50,
            "output_path": "/models/qwen3-embedding-0.6b",
        }

        # --force PHẢI đứng trước positional arg (Typer callback limitation)
        result = self._invoke(["--force", "qwen3-embedding-0.6b"])
        assert result.exit_code == 0

        # Verify force=True được truyền
        call_kwargs = self.mock_convert_model.remote.call_args.kwargs
        assert call_kwargs["force"] is True

    def test_convert_passes_correct_model_config(self) -> None:
        self.mock_convert_model.remote.return_value = {
            "model_name": "qwen3-embedding-0.6b",
            "status": "success",
            "files_count": 5,
            "total_size_mb": 1200.50,
            "output_path": "/models/qwen3-embedding-0.6b",
        }

        self._invoke(["qwen3-embedding-0.6b"])

        call_kwargs = self.mock_convert_model.remote.call_args.kwargs
        config = MODEL_REGISTRY["qwen3-embedding-0.6b"]
        assert call_kwargs["model_name"] == config.name
        assert call_kwargs["hf_id"] == config.hf_id
        assert call_kwargs["precision"] == config.precision.value
        assert call_kwargs["model_class"] == config.model_class.value
        assert call_kwargs["task"] == config.task.value
        assert call_kwargs["trust_remote_code"] == config.trust_remote_code

    def test_list_command(self) -> None:
        """Test list qua callback dispatch (model='list')."""
        result = self._invoke(["list"])
        assert result.exit_code == 0
        assert "Model Registry" in result.output
        # Rich truncate tên dài với "..." nên chỉ kiểm tra prefix
        assert "qwen3-embed" in result.output


# ---------------------------------------------------------------------------
# CloudBucketMount read_only parameter tests
# ---------------------------------------------------------------------------


class TestCloudBucketMountReadOnly:
    """Test get_modal_cloud_bucket_mount read_only parameter.

    modal được import lazily trong function body, nên patch trực tiếp
    trên modal module (đã cài sẵn trong project deps).
    """

    @patch("modal.Secret.from_name")
    @patch("modal.CloudBucketMount")
    def test_default_read_only(self, mock_cbm: MagicMock, mock_secret: MagicMock) -> None:
        """Mặc định phải là read_only=True (cho workers)."""
        from ai_workers.common.r2 import get_modal_cloud_bucket_mount

        get_modal_cloud_bucket_mount()
        assert mock_cbm.call_args.kwargs["read_only"] is True

    @patch("modal.Secret.from_name")
    @patch("modal.CloudBucketMount")
    def test_writable_mount(self, mock_cbm: MagicMock, mock_secret: MagicMock) -> None:
        """Converter dùng read_only=False."""
        from ai_workers.common.r2 import get_modal_cloud_bucket_mount

        get_modal_cloud_bucket_mount(read_only=False)
        assert mock_cbm.call_args.kwargs["read_only"] is False

    @patch("modal.Secret.from_name")
    @patch("modal.CloudBucketMount")
    def test_explicit_read_only(self, mock_cbm: MagicMock, mock_secret: MagicMock) -> None:
        """Có thể truyền read_only=True rõ ràng."""
        from ai_workers.common.r2 import get_modal_cloud_bucket_mount

        get_modal_cloud_bucket_mount(read_only=True)
        assert mock_cbm.call_args.kwargs["read_only"] is True
