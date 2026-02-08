"""R2 storage helpers for model weight management.

Uses boto3 S3-compatible API to interact with Cloudflare R2.
Also provides Modal CloudBucketMount configuration for workers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class R2Config:
    """Cloudflare R2 configuration."""

    bucket_name: str = "ai-workers-models"
    account_id: str = ""
    access_key_id: str = ""
    secret_access_key: str = ""

    @property
    def endpoint_url(self) -> str:
        return f"https://{self.account_id}.r2.cloudflarestorage.com"

    @classmethod
    def from_env(cls) -> R2Config:
        """Load R2 config from environment variables (Doppler/Infisical)."""
        return cls(
            bucket_name=os.getenv("R2_BUCKET_NAME", "ai-workers-models"),
            account_id=os.getenv("CF_ACCOUNT_ID", ""),
            access_key_id=os.getenv("CF_R2_ACCESS_KEY", ""),
            secret_access_key=os.getenv("CF_R2_SECRET_KEY", ""),
        )

    def validate(self) -> None:
        """Raise ValueError if required fields are missing."""
        missing = []
        if not self.account_id:
            missing.append("CF_ACCOUNT_ID")
        if not self.access_key_id:
            missing.append("CF_R2_ACCESS_KEY")
        if not self.secret_access_key:
            missing.append("CF_R2_SECRET_KEY")
        if missing:
            msg = f"Missing R2 environment variables: {', '.join(missing)}"
            raise ValueError(msg)


def get_s3_client(r2_config: R2Config):
    """Create a boto3 S3 client configured for Cloudflare R2."""
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=r2_config.endpoint_url,
        aws_access_key_id=r2_config.access_key_id,
        aws_secret_access_key=r2_config.secret_access_key,
        region_name="auto",
    )


def upload_directory(
    local_dir: Path,
    r2_prefix: str,
    r2_config: R2Config,
) -> int:
    """Upload a local directory to R2.

    Args:
        local_dir: Local directory containing model files.
        r2_prefix: R2 key prefix (e.g. "qwen3-embedding-0.6b").
        r2_config: R2 configuration.

    Returns:
        Number of files uploaded.
    """
    r2_config.validate()
    s3 = get_s3_client(r2_config)
    count = 0

    for file_path in sorted(local_dir.rglob("*")):
        if not file_path.is_file():
            continue
        relative = file_path.relative_to(local_dir)
        key = f"{r2_prefix}/{relative.as_posix()}"
        size_mb = file_path.stat().st_size / (1024**2)
        logger.info(
            f"Uploading {relative} ({size_mb:.1f} MB) -> s3://{r2_config.bucket_name}/{key}"
        )
        s3.upload_file(str(file_path), r2_config.bucket_name, key)
        count += 1

    logger.info(f"Uploaded {count} files to r2://{r2_config.bucket_name}/{r2_prefix}/")
    return count


def get_modal_cloud_bucket_mount(r2_config: R2Config | None = None):
    """Create a Modal CloudBucketMount for R2.

    Used inside Modal worker definitions to mount model weights.
    """
    import modal

    if r2_config is None:
        r2_config = R2Config.from_env()

    return modal.CloudBucketMount(
        bucket_name=r2_config.bucket_name,
        bucket_endpoint_url=r2_config.endpoint_url,
        secret=modal.Secret.from_name("r2-credentials"),
        read_only=True,
    )
