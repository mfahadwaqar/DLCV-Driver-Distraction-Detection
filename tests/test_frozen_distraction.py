import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from src.data.manifest import load_manifest
from src.data.splits import saved_splits
from src.eval.frozen_distraction import evaluate, inspect_inputs, sha256
from src.models.baseline import build_baseline


@pytest.fixture
def frozen_fixture(tmp_path):
    # Artificial pixels and random weights exercise software, not research accuracy.
    torch.set_num_threads(2)
    root = tmp_path
    images = root / "data/prepared"
    images.mkdir(parents=True)
    manifest = images / "manifest.csv"
    classes = [0, 1, 2, 3, 5]
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "subject_id", "dataset", "label_distraction", "label_drowsiness"])
        for subject in range(3):
            for label in classes:
                name = f"p{subject}_{label}.png"
                pixels = np.full((64, 64, 3), label * 30, dtype=np.uint8)
                Image.fromarray(pixels).save(images / name)
                writer.writerow([name, f"p{subject}", "state_farm", label, -1])
    splits = root / "data/splits.json"
    parts = saved_splits(load_manifest(manifest), splits)
    model_dir = root / "results/fixture/model"
    model_dir.mkdir(parents=True)
    torch.save(build_baseline(5, False, "resnet18").state_dict(), model_dir / "best.pt")
    config = {
        "model": "resnet18",
        "image_size": 64,
        "class_ids": classes,
        "label_key": "label_distraction",
        "evaluate_test": False,
    }
    (model_dir / "config.yaml").write_text(yaml.safe_dump(config))
    summary = {
        "training_performed": True,
        "fast": False,
        "test_evaluated": False,
        "model": "resnet18",
        "best_epoch": 1,
        "macro_f1": {"val": 0.5},
    }
    (model_dir / "summary.json").write_text(json.dumps(summary))
    (model_dir / "history.json").write_text(json.dumps([{"epoch": 1, "val_macro_f1": 0.5}]))
    (model_dir / "used_splits.json").write_text(json.dumps(parts))
    selection = {
        "experiment": "fixture",
        "model_dir": "results/fixture/model",
        **{key: config[key] for key in ("model", "image_size", "class_ids")},
        "checkpoint_sha256": sha256(model_dir / "best.pt"),
        "manifest": "data/prepared/manifest.csv",
        "manifest_sha256": sha256(manifest),
        "splits": "data/splits.json",
        "split_sha256": sha256(splits),
        "split_fingerprint": json.loads(splits.read_text())["fingerprint"],
        "test_samples": 5,
        "test_subjects": 1,
        "selection_metric": "validation_macro_f1",
        "validation_macro_f1": 0.5,
        "best_epoch": 1,
    }
    declaration = root / "selection.json"
    declaration.write_text(json.dumps(selection))
    return root, declaration, parts


def test_check_only_never_decodes_images_or_changes_files(frozen_fixture, monkeypatch):
    root, selection, _ = frozen_fixture
    before = {str(p): sha256(p) for p in root.rglob("*") if p.is_file()}
    monkeypatch.setattr(Image, "open", lambda *_a, **_k: pytest.fail("Check-only decoded image"))
    plan = inspect_inputs(root, selection)
    assert plan["ready"] and plan["test_samples"] == 5
    assert not plan["training_performed"] and not plan["inference_performed"]
    assert before == {str(p): sha256(p) for p in root.rglob("*") if p.is_file()}


def test_actual_checkpoint_test_only_inference_preserves_weights(frozen_fixture, monkeypatch):
    root, selection, parts = frozen_fixture
    allowed = {(root / "data/prepared" / row["path"]).resolve() for row in parts["test"]}
    opened = []
    original_open = Image.open

    def guard(path, *args, **kwargs):
        assert Path(path).resolve() in allowed, "Non-test image was accessed"
        opened.append(Path(path).resolve())
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Image, "open", guard)
    monkeypatch.setattr(torch.optim, "Adam", lambda *_a, **_k: pytest.fail("Optimizer created"))
    monkeypatch.setattr(torch.optim, "AdamW", lambda *_a, **_k: pytest.fail("Optimizer created"))
    original_train = torch.nn.Module.train

    def inference_only(self, mode=True):
        assert mode is False, "Model entered training mode"
        return original_train(self, mode)

    monkeypatch.setattr(torch.nn.Module, "train", inference_only)
    checkpoint = root / "results/fixture/model/best.pt"
    before = sha256(checkpoint)
    output = root / "test-output"
    summary = evaluate(root, output, batch_size=2, selection_path=selection)
    assert set(opened) == allowed and len(opened) == 5
    assert sha256(checkpoint) == before
    assert summary["training_performed"] is False and summary["test_evaluated"] is True
    report = json.loads((output / "test.json").read_text())
    assert report["samples"] == 5 and report["split"] == "test"
    assert sum(map(sum, report["confusion_matrix"])) == 5
    with (output / "test_predictions.csv").open(newline="") as stream:
        predictions = list(csv.DictReader(stream))
    assert [r["path"] for r in predictions] == [r["path"] for r in parts["test"]]
    assert (output / "frozen_checkpoint.json").exists()
    assert (output / "evaluation_provenance.json").exists()
    with pytest.raises(ValueError, match="preserve"):
        evaluate(root, output, selection_path=selection)


@pytest.mark.parametrize("field", ["checkpoint_sha256", "manifest_sha256", "split_sha256"])
def test_hash_drift_rejected_before_inference(frozen_fixture, field):
    root, path, _ = frozen_fixture
    selection = json.loads(path.read_text())
    selection[field] = "0" * 64
    path.write_text(json.dumps(selection))
    output = root / "output"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        evaluate(root, output, selection_path=path)
    assert not output.exists()


@pytest.mark.parametrize("change", ["model", "history", "used_splits", "selection_metric"])
def test_rejects_reselection_and_changed_configuration(frozen_fixture, change):
    root, path, parts = frozen_fixture
    model_dir = root / "results/fixture/model"
    if change == "model":
        config = yaml.safe_load((model_dir / "config.yaml").read_text())
        config["image_size"] = 224
        (model_dir / "config.yaml").write_text(yaml.safe_dump(config))
    elif change == "history":
        (model_dir / "history.json").write_text(json.dumps([{"epoch": 2, "val_macro_f1": 0.9}]))
    elif change == "used_splits":
        parts["test"] = parts["val"]
        (model_dir / "used_splits.json").write_text(json.dumps(parts))
    else:
        selection = json.loads(path.read_text())
        selection["selection_metric"] = "test_accuracy"
        path.write_text(json.dumps(selection))
    with pytest.raises(ValueError):
        inspect_inputs(root, path)


def test_missing_test_image_is_actionable(frozen_fixture):
    root, path, parts = frozen_fixture
    (root / "data/prepared" / parts["test"][0]["path"]).unlink()
    with pytest.raises(FileNotFoundError, match="Missing test image"):
        inspect_inputs(root, path)


def test_testing_notebook_is_readable_and_disabled_by_default():
    notebook = json.loads(Path("notebooks/driver_monitoring_testing.ipynb").read_text())
    code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert "RUN_TESTING = False" in code[0] and 'DEVICE = "cpu"' in code[0]
    for index, source in enumerate(code):
        compile(source, f"testing-cell-{index}", "exec")
    assert "RUN_TRAINING" not in "\n".join(code)
