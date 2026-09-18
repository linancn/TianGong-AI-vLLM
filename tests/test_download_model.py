import hashlib
import importlib.util
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

SPEC = importlib.util.spec_from_file_location(
    "download_model", Path(__file__).resolve().parents[1] / "scripts/download_model.py"
)
download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download)


def entry(data=b"model weights", name="weights.bin"):
    return {"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def test_verification_rejects_same_size_corruption(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"model weights")
    assert download.verified(path, entry())
    path.write_bytes(b"wrong weights")
    assert not download.verified(path, entry())


def test_verify_only_never_downloads(tmp_path):
    with patch.object(download.subprocess, "run") as run:
        with pytest.raises(RuntimeError, match="Missing or corrupt"):
            download.fetch(tmp_path, {}, entry(), True)
        run.assert_not_called()


def test_failed_download_preserves_existing_file(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"old file")
    with patch.object(
        download.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "curl")
    ):
        with pytest.raises(subprocess.CalledProcessError):
            download.fetch(tmp_path, {"model_id": "owner/model", "revision": "abc"}, entry(), False)
    assert path.read_bytes() == b"old file"


def test_hash_failure_never_publishes_partial_file(tmp_path):
    def corrupt(*args, **kwargs):
        (tmp_path / "weights.bin.part").write_bytes(b"bad download")

    with patch.object(download.subprocess, "run", side_effect=corrupt):
        with pytest.raises(RuntimeError, match="SHA256 mismatch"):
            download.fetch(tmp_path, {"model_id": "owner/model", "revision": "abc"}, entry(), False)
    assert not (tmp_path / "weights.bin").exists()
    assert not (tmp_path / "weights.bin.part").exists()


def test_manifest_cannot_escape_model_directory(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        download.fetch(tmp_path, {}, entry(name="../outside"), False)
