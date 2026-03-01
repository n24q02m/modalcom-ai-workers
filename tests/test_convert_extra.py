"""Tests for cli/convert.py covering previously uncovered lines.

Covers: model=None (48-51), model='all' (58-72), AuthError (128-130),
generic exception (131-136), and unknown status (124-126).
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
import typer.testing

from ai_workers.cli.convert import app


class TestConvertCLIExtra:
    """Extra tests for convert CLI covering uncovered branches."""

    @pytest.fixture(autouse=True)
    def _setup_mocks(self):
        """Pre-populate sys.modules with mock converter module."""
        self.mock_convert_model = MagicMock()
        mock_converter_module = MagicMock()
        mock_converter_module.convert_model = self.mock_convert_model

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
        return runner.invoke(app, args)

    def test_model_none_shows_error(self) -> None:
        """Calling convert with no model arg should print error and exit 1 (lines 48-51)."""
        result = self._invoke([])
        assert result.exit_code != 0
        assert "Cần chỉ định" in result.output or "help" in result.output.lower()

    def test_convert_all_success(self) -> None:
        """model='all' should convert all models and exit 0 (lines 58-72)."""
        self.mock_convert_model.remote.return_value = {
            "model_name": "test",
            "status": "success",
            "files_count": 3,
            "total_size_mb": 500.0,
            "output_path": "/models/test",
        }
        result = self._invoke(["all"])
        assert result.exit_code == 0
        assert "thành công" in result.output.lower() or "success" in result.output.lower()
        # Should have called remote once per model in registry
        assert self.mock_convert_model.remote.call_count > 0

    def test_convert_all_partial_failure_exits_nonzero(self) -> None:
        """model='all' with some failures should exit 1 (lines 66-70)."""
        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Modal error")
            return {
                "model_name": kwargs.get("model_name", "test"),
                "status": "success",
                "files_count": 1,
                "total_size_mb": 100.0,
                "output_path": "/models/test",
            }

        self.mock_convert_model.remote.side_effect = side_effect
        result = self._invoke(["all"])
        assert result.exit_code != 0

    def test_convert_unknown_status_exits_nonzero(self) -> None:
        """Unknown status in result should print error and exit 1 (lines 124-126)."""
        self.mock_convert_model.remote.return_value = {
            "model_name": "qwen3-embedding-0.6b",
            "status": "weird_status",
        }
        result = self._invoke(["qwen3-embedding-0.6b"])
        assert result.exit_code != 0
        assert "weird_status" in result.output or "không xác định" in result.output

    def test_convert_auth_error_exits_nonzero(self) -> None:
        """modal.exception.AuthError should show auth error message (lines 128-130)."""
        import modal

        auth_error_cls = modal.exception.AuthError
        self.mock_convert_model.remote.side_effect = auth_error_cls("Not authenticated")
        result = self._invoke(["qwen3-embedding-0.6b"])
        assert result.exit_code != 0
        assert "xác thực" in result.output or "modal token" in result.output.lower()

    def test_convert_generic_exception_exits_nonzero(self) -> None:
        """Generic exception should show failure message (lines 131-136)."""
        self.mock_convert_model.remote.side_effect = ConnectionError("Network error")
        result = self._invoke(["qwen3-embedding-0.6b"])
        assert result.exit_code != 0
        assert "THẤT BẠI" in result.output or "Network error" in result.output

    def test_list_command_shows_registry(self) -> None:
        """list subcommand should show model registry table."""
        result = self._invoke(["list"])
        assert result.exit_code == 0
        assert "Model Registry" in result.output
