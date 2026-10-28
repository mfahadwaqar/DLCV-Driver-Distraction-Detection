import csv
import json

import pytest

from scripts.analyze_rt_bene_errors import analyze


def test_subject_errors_reproduce_confusion_and_reject_misaligned_predictions(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "test.json").write_text(
        json.dumps(
            {
                "task": "eye-state",
                "dataset_scope": ["rt_bene"],
                "samples": 3,
                "confusion_matrix": [[1, 0], [1, 1]],
            }
        )
    )
    (artifacts / "used_splits.json").write_text(
        json.dumps(
            {
                "test": [
                    {"subject_id": "rt_bene:s001", "path": "s001/a.png", "label_drowsiness": 0},
                    {"subject_id": "rt_bene:s001", "path": "s001/b.png", "label_drowsiness": 1},
                    {"subject_id": "rt_bene:s002", "path": "s002/c.png", "label_drowsiness": 1},
                ]
            }
        )
    )
    with (artifacts / "test_predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerows(
            [
                ["label", "prediction", "p_0", "p_1"],
                [0, 0, 0.9, 0.1],
                [1, 0, 0.8, 0.2],
                [1, 1, 0.1, 0.9],
            ]
        )
    result = analyze(artifacts, tmp_path / "analysis")
    assert result["false_negatives"] == 1
    assert result["subjects"][0]["closed_recall"] == 0
    assert result["subjects"][1]["closed_recall"] == 1
    assert (tmp_path / "analysis/false_negatives.csv").is_file()

    with (artifacts / "test_predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerows(
            [
                ["label", "prediction", "p_0", "p_1"],
                [1, 0, 0.9, 0.1],
                [0, 0, 0.8, 0.2],
                [1, 1, 0.1, 0.9],
            ]
        )
    with pytest.raises(ValueError, match="order"):
        analyze(artifacts, tmp_path / "analysis")
