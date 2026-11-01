import csv
import json

import pytest
from PIL import Image

from scripts.analyze_distraction_validation import CLASSES, analyze
from src.data.schema import DISTRACTION


def fixture(tmp_path):
    artifacts = tmp_path / "artifacts"
    images = tmp_path / "images"
    artifacts.mkdir()
    images.mkdir()
    rows = [
        {
            "path": "safe.jpg",
            "dataset": "state_farm",
            "subject_id": "state_farm:p1",
            "label_distraction": 0,
            "metadata": {"source_class": "c0"},
        },
        {
            "path": "passenger.jpg",
            "dataset": "state_farm",
            "subject_id": "state_farm:p2",
            "label_distraction": 5,
            "metadata": {"source_class": "c9"},
        },
    ]
    for row in rows:
        Image.new("RGB", (20, 20)).save(images / row["path"])
    matrix = [[0] * 5 for _ in range(5)]
    matrix[0][4] = matrix[4][3] = 1
    report = {
        "split": "val",
        "dataset_scope": ["state_farm"],
        "samples": 2,
        "class_order": [DISTRACTION[i] for i in CLASSES],
        "confusion_matrix": matrix,
    }
    (artifacts / "val.json").write_text(json.dumps(report))
    (artifacts / "used_splits.json").write_text(json.dumps({"val": rows}))
    with (artifacts / "val_predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["label", "prediction", "p_5", "p_3"])
        writer.writeheader()
        writer.writerows(
            [{"label": 0, "prediction": 5, "p_5": 0.8}, {"label": 5, "prediction": 3, "p_3": 0.7}]
        )
    return artifacts, images


def test_validation_error_counts_and_local_contact_sheets(tmp_path):
    artifacts, images = fixture(tmp_path)
    result = analyze(artifacts, images, tmp_path / "out")
    assert result["samples"] == result["errors"] == 2
    assert result["source_mapping_verified"]
    assert (tmp_path / "out/validation_0_as_5.jpg").is_file()
    assert (tmp_path / "out/validation_5_as_3.jpg").is_file()


def test_mismatched_prediction_labels_cannot_be_assigned_to_images(tmp_path):
    artifacts, images = fixture(tmp_path)
    path = artifacts / "val_predictions.csv"
    path.write_text(path.read_text().replace("0,5,", "1,5,"))
    with pytest.raises(ValueError, match="order/labels differ"):
        analyze(artifacts, images, tmp_path / "out")


def test_test_report_is_rejected_before_image_access(tmp_path):
    artifacts, images = fixture(tmp_path)
    path = artifacts / "val.json"
    report = json.loads(path.read_text())
    report["split"] = "test"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="validation metrics"):
        analyze(artifacts, images, tmp_path / "out")
