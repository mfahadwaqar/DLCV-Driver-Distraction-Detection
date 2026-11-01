import base64
import hashlib
import io
import json
import zipfile

import pytest

from src.train.notebook_backup import MIME, backup_payload, restore_notebook


def notebook(tmp_path, payload):
    path = tmp_path / "saved.ipynb"
    path.write_text(json.dumps({"cells": [{"outputs": [{"data": {MIME: payload}}]}]}))
    return path


def test_saved_notebook_restores_exact_checkpoint_and_preserves_existing(tmp_path):
    experiment = tmp_path / "remote" / "run1"
    (experiment / "model").mkdir(parents=True)
    (experiment / "model/best.pt").write_bytes(b"synthetic-checkpoint-backup")
    (experiment / "run_status.json").write_text('{"returncode":0}')
    payload = backup_payload(experiment)
    saved = notebook(tmp_path, payload)
    restored = restore_notebook(saved, tmp_path / "local")
    assert (restored[0] / "model/best.pt").read_bytes() == b"synthetic-checkpoint-backup"
    assert json.loads((restored[0] / "run_status.json").read_text())["returncode"] == 0
    with pytest.raises(FileExistsError, match="Preserving existing"):
        restore_notebook(saved, tmp_path / "local")


def test_corrupt_notebook_backup_fails_hash_verification(tmp_path):
    payload = {
        "experiment": "run1",
        "sha256": "wrong",
        "archive_base64": base64.b64encode(b"bad").decode(),
    }
    with pytest.raises(ValueError, match="SHA-256"):
        restore_notebook(notebook(tmp_path, payload), tmp_path / "local")


def test_escape_archive_is_rejected_before_extraction(tmp_path):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("run1/model/best.pt", b"test")
        archive.writestr("../escape.txt", b"bad")
    content = stream.getvalue()
    payload = {
        "experiment": "run1",
        "sha256": hashlib.sha256(content).hexdigest(),
        "archive_base64": base64.b64encode(content).decode(),
    }
    with pytest.raises(ValueError, match="Unsafe"):
        restore_notebook(notebook(tmp_path, payload), tmp_path / "local")
    assert not (tmp_path / "escape.txt").exists()
