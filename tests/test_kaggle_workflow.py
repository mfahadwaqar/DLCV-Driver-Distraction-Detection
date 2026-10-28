import base64
import csv
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from scripts import kaggle_workflow as workflow


def test_bundle_uses_only_committed_source_and_records_commit(tmp_path, monkeypatch):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("src/__init__.py", "")
    calls = []

    def fake_git(*args):
        calls.append(args)
        if args[0] == "status":
            return b""
        if args[0] == "rev-parse":
            return b"abc123\n"
        return archive.getvalue()

    monkeypatch.setattr(workflow, "git", fake_git)
    stage = tmp_path / "stage"
    assert workflow.prepare(stage) == "abc123"
    assert calls[-1] == ("archive", "--format=zip", "HEAD", "src", "configs")
    notebook = json.loads((stage / "run.ipynb").read_text())
    code = "".join(notebook["cells"][0]["source"])
    compile(code, "run.ipynb", "exec")
    payload = code.split("base64.b64decode(")[1].split(")")[0].strip("'")
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as stream:
        assert set(stream.namelist()) == {"src/__init__.py", "SOURCE_COMMIT"}
        assert stream.read("SOURCE_COMMIT") == b"abc123\n"
    metadata = json.loads((stage / "kernel-metadata.json").read_text())
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is False


def test_dirty_source_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, "git", lambda *args: b" M src/train.py")
    with pytest.raises(RuntimeError, match="Commit changes first"):
        workflow.prepare(tmp_path / "stage")


def test_phase1_requires_explicit_attached_data_path(tmp_path):
    with pytest.raises(ValueError, match="data-root"):
        workflow.prepare(tmp_path / "stage", task="phase1")


def test_rt_bene_bundle_fetches_official_data_in_private_internet_notebook(tmp_path, monkeypatch):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("src/__init__.py", "")
    monkeypatch.setattr(
        workflow,
        "git",
        lambda *args: (
            b""
            if args[0] == "status"
            else b"abc123\n"
            if args[0] == "rev-parse"
            else archive.getvalue()
        ),
    )
    stage = tmp_path / "stage"
    workflow.prepare(stage, task="rt-bene", fast=True, device="cuda")
    notebook = json.loads((stage / "run.ipynb").read_text())
    code = "".join(notebook["cells"][0]["source"])
    compile(code, "run.ipynb", "exec")
    assert "src.data.fetch_rt_bene" in code
    assert "src.data.rt_bene" in code
    assert "--manifest" in code
    metadata = json.loads((stage / "kernel-metadata.json").read_text())
    assert metadata["is_private"] is True
    assert metadata["enable_internet"] is True
    assert metadata["enable_gpu"] is True


def test_missing_auth_is_failure_even_if_cli_returns_zero(monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda path: True)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(
            a, 0, "Authentication required to call the Kaggle API.", ""
        ),
    )
    with pytest.raises(RuntimeError, match="local sign-in"):
        workflow.cli("kernels", "status", "owner/notebook")


def test_smoke_writes_real_counts_and_no_training_claim(tmp_path):
    input_root = tmp_path / "input"
    data = input_root / "competitions" / "state-farm"
    data.mkdir(parents=True)
    with (data / "driver_imgs_list.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerows(
            [
                ["subject", "classname", "img"],
                ["p1", "c0", "a.jpg"],
                ["p2", "c1", "b.jpg"],
            ]
        )
    output = tmp_path / "artifacts"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.train.kaggle_entry",
            "--fast",
            "--input",
            str(input_root),
            "--output",
            str(output),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    report = json.loads((output / "smoke.json").read_text())
    assert report["state_farm"]["rows"] == 2
    assert report["state_farm"]["subjects"] == 2
    assert report["training_performed"] is False
    assert (output / "environment.txt").stat().st_size > 0
