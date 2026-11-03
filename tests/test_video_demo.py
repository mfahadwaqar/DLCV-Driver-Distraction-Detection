import ast
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

import src.demo.run as demo
from src.demo.workflow import inspect_demo, record_development_usage, run_demo
from src.eval.frozen_distraction import sha256


def make_video(path, frames=6):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 64))
    assert writer.isOpened(), "Local test requires the MP4 encoder used by the demo"
    for i in range(frames):
        writer.write(np.full((64, 64, 3), 20 + i * 10, dtype=np.uint8))
    writer.release()


@pytest.fixture
def fake_predictors(monkeypatch):
    calls = {"label_drowsiness": 0, "label_distraction": 0}

    class Classifier:
        def __init__(self, folder, target, device):
            self.target = target

        def predict(self, rgb):
            calls[self.target] += 1
            return 1, 0.9

    class Cropper:
        def __init__(self, path):
            self.closed = False

        def crop(self, rgb, timestamp_ms):
            return None

        def close(self):
            self.closed = True

    monkeypatch.setattr(demo, "Classifier", Classifier)
    monkeypatch.setattr(demo, "EyeCropper", Cropper)
    return calls


def test_video_logs_unknown_eye_but_keeps_full_frame_distraction(tmp_path, fake_predictors):
    source = tmp_path / "source.mp4"
    make_video(source)
    args = SimpleNamespace(
        source=str(source),
        eye_checkpoint=tmp_path,
        distraction_checkpoint=tmp_path,
        face_model=tmp_path / "face.task",
        events=tmp_path / "events.jsonl",
        device="cpu",
        confidence=0.7,
        persist_seconds=0.08,
        max_gap=0.75,
        max_frames=6,
        no_display=True,
        frame_log=tmp_path / "frames.jsonl",
        output_video=tmp_path / "annotated.mp4",
        summary=tmp_path / "summary.json",
    )
    summary = demo.run(args)
    assert fake_predictors == {"label_drowsiness": 0, "label_distraction": 6}
    assert summary["processed_frames"] == 6 and summary["eye_coverage"] == 0
    assert summary["alert_starts"] == {"eye": 0, "distraction": 1}
    assert not summary["scoring_performed"] and not summary["training_performed"]
    records = [json.loads(line) for line in args.frame_log.read_text().splitlines()]
    assert [r["frame"] for r in records] == list(range(6))
    assert [r["timestamp"] for r in records] == [i / 25 for i in range(6)]
    assert all(r["predictions"]["eye"]["label"] is None for r in records)
    assert all(r["predictions"]["distraction"]["label"] == 1 for r in records)
    events = [json.loads(line) for line in args.events.read_text().splitlines()]
    starts = [r for r in events if r["event"] == "start"]
    assert len(starts) == 1 and starts[0]["timestamp"] >= 0.08
    assert events[-1]["event"] == "run_end"
    assert args.output_video.stat().st_size > 0
    with pytest.raises(ValueError, match="preserve"):
        demo.run(args)


@pytest.fixture
def demo_fixture(tmp_path):
    root = tmp_path
    config_dir = root / "configs"
    config_dir.mkdir()
    for name in ("eye", "distraction"):
        folder = root / "results" / name
        folder.mkdir(parents=True)
        (folder / "best.pt").write_bytes(name.encode())
        config = {
            "model": "mobilenet_v3_small" if name == "eye" else "resnet18",
            "class_ids": [0, 1] if name == "eye" else [0, 1, 2, 3, 5],
            "image_size": 96 if name == "eye" else 224,
            "label_key": "label_drowsiness" if name == "eye" else "label_distraction",
        }
        (folder / "config.yaml").write_text(yaml.safe_dump(config))
    asset = root / "data/models/face.task"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"synthetic asset, not used for actual inference")
    source = root / "data/demo/development.mp4"
    make_video(source)
    selected = {
        "model_dir": "results/distraction",
        "model": "resnet18",
        "image_size": 224,
        "class_ids": [0, 1, 2, 3, 5],
        "checkpoint_sha256": sha256(root / "results/distraction/best.pt"),
    }
    (config_dir / "selected_distraction_checkpoint.json").write_text(json.dumps(selected))
    settings = {
        "eye_model_dir": "results/eye",
        "eye_model": "mobilenet_v3_small",
        "eye_image_size": 96,
        "eye_checkpoint_sha256": sha256(root / "results/eye/best.pt"),
        "face_model": "data/models/face.task",
        "face_model_sha256": sha256(asset),
        "development_source": "data/demo/development.mp4",
        "confidence": 0.7,
        "persist_seconds": 0.08,
        "max_gap": 0.75,
    }
    (config_dir / "video_demo.json").write_text(json.dumps(settings))
    return root


def test_demo_inspection_checks_assets_without_predicting(demo_fixture, monkeypatch):
    monkeypatch.setattr(demo, "Classifier", lambda *_a: pytest.fail("Inspection ran classifier"))
    plan = inspect_demo(demo_fixture)
    assert plan["ready"] and plan["source_frames"] == 6
    assert not plan["training_performed"] and not plan["scoring_performed"]
    assert not (demo_fixture / "data/demo/development_usage.json").exists()


@pytest.mark.parametrize("name", ["eye", "distraction"])
def test_demo_rejects_checkpoint_changes(demo_fixture, name):
    (demo_fixture / "results" / name / "best.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        inspect_demo(demo_fixture)


def test_dmd_source_is_rejected_and_usage_is_preserved(demo_fixture):
    with pytest.raises(ValueError, match="keep DMD held out"):
        inspect_demo(demo_fixture, demo_fixture / "data/dmd/example.mp4")
    plan = inspect_demo(demo_fixture)
    record_development_usage(demo_fixture, plan)
    record_development_usage(demo_fixture, plan)
    records = json.loads((demo_fixture / "data/demo/development_usage.json").read_text())
    assert len(records) == 1 and records[0]["role"] == "qualitative_development_demo"
    assert records[0]["identity_status"] == "not established here"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg not installed")
def test_demo_workflow_creates_playable_preview_and_preserves_inputs(demo_fixture, fake_predictors):
    root = demo_fixture
    source = root / "data/demo/development.mp4"
    before = sha256(source)
    summary = run_demo(root, root / "output", max_frames=4)
    assert summary["demo_only"] and not summary["scoring_performed"]
    assert summary["processed_frames"] == summary["decoded_preview_frames"] == 4
    assert Path(summary["preview"]).stat().st_size > 0
    assert sha256(source) == before
    assert (root / "output/input_provenance.json").is_file()
    with pytest.raises(ValueError, match="preserve"):
        run_demo(root, root / "output")


def test_video_notebook_compiles_and_has_explicit_boolean_control():
    notebook = json.loads(Path("notebooks/driver_monitoring_video_demo.ipynb").read_text())
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    controls = [
        node.value
        for node in ast.parse(cells[0]).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "RUN_DEMO" for target in node.targets)
    ]
    assert len(controls) == 1 and isinstance(controls[0], ast.Constant)
    assert type(controls[0].value) is bool
    for index, source in enumerate(cells):
        compile(source, f"video-cell-{index}", "exec")
