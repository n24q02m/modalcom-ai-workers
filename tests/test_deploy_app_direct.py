import subprocess
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from ai_workers.cli.deploy import _deploy_app, _deploy_module, app

runner = CliRunner()


def test_deploy_app_success():
    """Test _deploy_app success path."""
    with patch("ai_workers.cli.deploy.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        _deploy_app("mod", "app_var")
        mock_run.assert_called_once()


def test_deploy_app_called_process_error():
    """Test _deploy_app handles CalledProcessError."""
    with patch("ai_workers.cli.deploy.subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(returncode=1, cmd="modal deploy")
        # _deploy_app raises typer.Exit(code=1)
        with pytest.raises(typer.Exit) as exc:
            _deploy_app("mod", "app_var")
        assert exc.value.exit_code == 1


def test_deploy_app_file_not_found():
    """Test _deploy_app handles FileNotFoundError."""
    with patch("ai_workers.cli.deploy.subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError()
        with pytest.raises(typer.Exit) as exc:
            _deploy_app("mod", "app_var")
        assert exc.value.exit_code == 1


def test_deploy_no_worker_error():
    """Test deploy callback handles missing worker argument."""
    # To hit line 61-62: worker is None and all_workers is False
    result = runner.invoke(app, ["--dry-run"])
    assert result.exit_code == 1
    assert "Error: specify a model name or use --all" in result.output


def test_deploy_module_called_process_error():
    """Test _deploy_module handles CalledProcessError."""
    with patch("ai_workers.cli.deploy.subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(returncode=1, cmd="modal deploy")
        with pytest.raises(typer.Exit) as exc:
            _deploy_module("mod")
        assert exc.value.exit_code == 1


def test_deploy_module_file_not_found():
    """Test _deploy_module handles FileNotFoundError."""
    with patch("ai_workers.cli.deploy.subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError()
        with pytest.raises(typer.Exit) as exc:
            _deploy_module("mod")
        assert exc.value.exit_code == 1
