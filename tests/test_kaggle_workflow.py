import base64
import csv
import io
import json
import subprocess
import sys
import zipfile
import zlib
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
    assert "/kaggle/temp/rt-bene/source" in code
    assert "/kaggle/temp/rt-bene/prepared" in code
    metadata = json.loads((stage / "kernel-metadata.json").read_text())
    assert metadata["is_private"] is True
    assert metadata["enable_internet"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["competition_sources"] == []


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


def test_state_farm_bundle_uses_competition_input_and_distraction_config(tmp_path, monkeypatch):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("src/__init__.py", "")
    monkeypatch.setattr(
        workflow,
        "git",
        lambda *args: (
            b""
            if args[0] == "status"
            else b"abc123"
            if args[0] == "rev-parse"
            else archive.getvalue()
        ),
    )
    stage = tmp_path / "stage"
    workflow.prepare(stage, task="state-farm", fast=True, device="cuda")
    notebook = json.loads((stage / "run.ipynb").read_text())
    code = "".join(notebook["cells"][0]["source"])
    compile(code, "run.ipynb", "exec")
    assert "src.data.state_farm" in code
    assert "configs/state_farm.yaml" in code
    assert "/kaggle/input" in code
    metadata = json.loads((stage / "kernel-metadata.json").read_text())
    assert metadata["competition_sources"] == ["state-farm-distracted-driver-detection"]
    assert metadata["enable_internet"] is False


def test_pretrained_state_farm_bundles_saved_splits_and_enables_internet(tmp_path, monkeypatch):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("src/__init__.py", "")
        stream.writestr("configs/state_farm_pretrained.yaml", "pretrained: true\n")
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
    baseline = {
        "seed": 42,
        "fractions": [0.7, 0.15, 0.15],
        "fingerprint": "baseline",
        "splits": {
            name: [
                {
                    "path": "state-farm-distracted-driver-detection/imgs/train/c0/a.jpg",
                    "subject_id": f"state_farm:state_farm:{name}",
                    "dataset": "state_farm",
                    "label_distraction": 0,
                    "label_drowsiness": -1,
                    "metadata": {"source_class": "c0"},
                }
            ]
            for name in ("train", "val", "test")
        },
    }
    baseline_path = tmp_path / "state_farm_splits.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    monkeypatch.setattr(workflow, "BASELINE_SPLITS", baseline_path)

    stage = tmp_path / "stage"
    workflow.prepare(stage, task="state-farm-pretrained", device="cuda")
    notebook = json.loads((stage / "run.ipynb").read_text())
    code = "".join(notebook["cells"][0]["source"])
    compile(code, "run.ipynb", "exec")
    assert "configs/state_farm_pretrained.yaml" in code
    assert "baseline_splits.json" in code
    assert "/kaggle/working/artifacts/state_farm_pretrained" in code
    metadata = json.loads((stage / "kernel-metadata.json").read_text())
    assert metadata["enable_internet"] is True
    assert metadata["enable_gpu"] is True

    b64 = code.split("split_path.write_bytes(zlib.decompress(base64.b64decode(")[1].split(")")[0]
    embedded = json.loads(zlib.decompress(base64.b64decode(b64.strip("'"))))
    assert embedded == baseline
    assert embedded["splits"]["train"][0]["path"].startswith(
        "state-farm-distracted-driver-detection/"
    )


def test_baseline_split_loader_rejects_arbitrary_fields(tmp_path):
    path = tmp_path / "splits.json"
    path.write_text(json.dumps({"secret": "not allowed"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected baseline split document fields"):
        workflow.load_baseline_splits(path)
