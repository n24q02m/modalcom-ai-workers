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
    """Cloudflare R2 configuration.

    Environment variables:
        R2_ENDPOINT_URL:      Full R2 endpoint (e.g. https://<account>.r2.cloudflarestorage.com)
        R2_ACCESS_KEY_ID:     R2 API token access key
        R2_SECRET_ACCESS_KEY: R2 API token secret key
        R2_BUCKET_NAME:       Bucket name (default: ai-workers-models)
    """

    bucket_name: str = "ai-workers-models"
    endpoint_url: str = ""
    access_key_id: str = ""
    secret_access_key: str = ""

    @classmethod
    def from_env(cls) -> R2Config:
        """Load R2 config from environment variables."""
        return cls(
            bucket_name=os.getenv("R2_BUCKET_NAME", "ai-workers-models"),
            endpoint_url=os.getenv("R2_ENDPOINT_URL", ""),
            access_key_id=os.getenv("R2_ACCESS_KEY_ID", ""),
            secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", ""),
        )

    def validate(self) -> None:
        """Raise ValueError if required fields are missing."""
        missing = []
        if not self.endpoint_url:
            missing.append("R2_ENDPOINT_URL")
        if not self.access_key_id:
            missing.append("R2_ACCESS_KEY_ID")
        if not self.secret_access_key:
            missing.append("R2_SECRET_ACCESS_KEY")
        if missing:
            msg = f"Missing R2 environment variables: {', '.join(missing)}"
            raise ValueError(msg)


def get_s3_client(r2_config: R2Config):
    """Create a boto3 S3 client configured for Cloudflare R2."""
    r2_config.validate()

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


def get_modal_cloud_bucket_mount(
    r2_config: R2Config | None = None,
    *,
    bucket_name: str = "",
    bucket_endpoint_url: str = "",
    read_only: bool = True,
):
    """Tạo Modal CloudBucketMount cho R2.

    Dùng trong Modal worker definitions để mount model weights.
    Workers serving dùng read_only=True (mặc định).
    Converter dùng read_only=False để ghi model lên R2.

    Modal Secret ``r2-credentials`` BẮT BUỘC chứa:
      - AWS_ACCESS_KEY_ID:     R2 access key (tên S3-compatible)
      - AWS_SECRET_ACCESS_KEY: R2 secret key (tên S3-compatible)

    Nếu read_only=False, R2 API token cần có quyền write + list.

    Args:
        r2_config: Config tùy chọn cho bucket_name/endpoint overrides.
        bucket_name: Ghi đè bucket name (ưu tiên hơn r2_config).
        bucket_endpoint_url: Ghi đè endpoint URL (ưu tiên hơn r2_config).
        read_only: Mount chỉ đọc (True) hoặc đọc-ghi (False).
    """
    import modal

    if r2_config is None:
        r2_config = R2Config.from_env()

    _bucket = bucket_name or r2_config.bucket_name
    _endpoint = bucket_endpoint_url or r2_config.endpoint_url

    return modal.CloudBucketMount(
        bucket_name=_bucket,
        bucket_endpoint_url=_endpoint,
        secret=modal.Secret.from_name("r2-credentials"),
        read_only=read_only,
    )
