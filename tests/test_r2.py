"""Tests for R2 configuration."""

import os
from unittest.mock import patch

import pytest

from ai_workers.common.r2 import R2Config


class TestR2Config:
    """Test the R2Config class and its validate method."""

    def test_validate_success(self) -> None:
        """Validate should pass with all required fields."""
        config = R2Config(
            account_id="test-account",
            access_key_id="test-key",
            secret_access_key="test-secret",
        )
        config.validate()  # Should not raise

    @pytest.mark.parametrize(
        "account_id, access_key_id, secret_access_key, missing_fields",
        [
            ("", "k", "s", ["CF_ACCOUNT_ID"]),
            ("a", "", "s", ["CF_R2_ACCESS_KEY"]),
            ("a", "k", "", ["CF_R2_SECRET_KEY"]),
            ("", "", "s", ["CF_ACCOUNT_ID", "CF_R2_ACCESS_KEY"]),
            ("", "k", "", ["CF_ACCOUNT_ID", "CF_R2_SECRET_KEY"]),
            ("a", "", "", ["CF_R2_ACCESS_KEY", "CF_R2_SECRET_KEY"]),
            ("", "", "", ["CF_ACCOUNT_ID", "CF_R2_ACCESS_KEY", "CF_R2_SECRET_KEY"]),
        ],
    )
    def test_validate_missing_fields(
        self,
        account_id: str,
        access_key_id: str,
        secret_access_key: str,
        missing_fields: list[str],
    ) -> None:
        """Validate should raise ValueError with correct missing fields."""
        config = R2Config(
            account_id=account_id,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
        )
        with pytest.raises(ValueError) as exc:
            config.validate()

        msg = str(exc.value)
        assert "Missing R2 environment variables" in msg
        for field in missing_fields:
            assert field in msg

    def test_from_env(self) -> None:
        """Test loading config from environment variables."""
        env_vars = {
            "CF_ACCOUNT_ID": "env-account",
            "CF_R2_ACCESS_KEY": "env-key",
            "CF_R2_SECRET_KEY": "env-secret",
            "R2_BUCKET_NAME": "env-bucket",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = R2Config.from_env()
            assert config.account_id == "env-account"
            assert config.access_key_id == "env-key"
            assert config.secret_access_key == "env-secret"
            assert config.bucket_name == "env-bucket"

    def test_from_env_defaults(self) -> None:
        """Test default values when env vars are missing."""
        with patch.dict(os.environ, {}, clear=True):
            config = R2Config.from_env()
            assert config.account_id == ""
            assert config.access_key_id == ""
            assert config.secret_access_key == ""
            assert config.bucket_name == "ai-workers-models"  # Default value
