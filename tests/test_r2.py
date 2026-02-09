import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai_workers.common.r2 import R2Config, upload_directory


@pytest.fixture
def temp_dirs():
    # Create a temporary directory for the "upload" source
    upload_dir = tempfile.mkdtemp()
    # Create a temporary directory for "sensitive" files outside the upload dir
    sensitive_dir = tempfile.mkdtemp()

    yield Path(upload_dir), Path(sensitive_dir)

    shutil.rmtree(upload_dir)
    shutil.rmtree(sensitive_dir)


def test_symlink_upload_vulnerability(temp_dirs):
    upload_dir, sensitive_dir = temp_dirs

    # 1. Create a sensitive file outside the upload directory
    sensitive_file = sensitive_dir / "passwd"
    sensitive_file.write_text("root:x:0:0:root:/root:/bin/bash")

    # 2. Create a symlink in the upload directory pointing to the sensitive file
    symlink = upload_dir / "passwd_link"
    try:
        os.symlink(sensitive_file, symlink)
    except OSError:
        pytest.skip("Symlinks not supported on this platform")

    # 3. Create a normal file in the upload directory
    normal_file = upload_dir / "normal.txt"
    normal_file.write_text("hello world")

    # 4. Mock R2Config and boto3
    r2_config = MagicMock(spec=R2Config)
    r2_config.bucket_name = "test-bucket"
    r2_config.endpoint_url = "https://example.com"
    r2_config.access_key_id = "test"
    r2_config.secret_access_key = "test"

    mock_s3 = MagicMock()

    with patch("ai_workers.common.r2.get_s3_client", return_value=mock_s3):
        # 5. Call upload_directory
        upload_directory(upload_dir, "prefix", r2_config)

    # 6. Verify what was uploaded
    uploaded_files = []
    for call in mock_s3.upload_file.call_args_list:
        args, _ = call
        src_path = args[0]
        uploaded_files.append(Path(src_path).name)

    # The fix ensures symlinks are skipped
    print(f"Uploaded files: {uploaded_files}")
    assert "passwd_link" not in uploaded_files
    assert "passwd" not in uploaded_files
    assert sensitive_file.name not in uploaded_files

    # Normal files should still be uploaded
    assert "normal.txt" in uploaded_files
