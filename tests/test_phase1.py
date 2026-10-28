import csv
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.manifest import load_manifest
from src.data.mrl import ImageDataset, index_mrl, parse_filename
from src.data.splits import assert_no_leakage, fast_subset, make_splits, saved_splits
from src.eval.metrics import classification_metrics, save_metrics
from src.eval.rules import score, tune_thresholds
from src.models.baseline import build_baseline
from src.models.rules import (
    TemporalRules,
    Thresholds,
    eye_aspect_ratio,
    mouth_aspect_ratio,
    pitch_degrees,
)
from src.train.losses import masked_cross_entropy
from src.train.phase1 import encode_labels, image_transform, seed_everything, train


def records():
    return [
        dict(
            path=f"{subject}/{label}.png",
            subject_id=f"mock:{subject}",
            dataset="mock",
            label_distraction=-1,
            label_drowsiness=label,
            metadata={"synthetic": True},
        )
        for subject in range(8)
        for label in (0, 1)
    ]


@pytest.mark.parametrize("eye,label", [(0, 1), (1, 0)])
def test_mrl_label_mapping_and_flags(eye, label):
    row = parse_filename(f"s0002_00001_0_1_{eye}_2_0_01.png")
    assert row["subject_id"] == "mrl:s0002"
    assert row["label_distraction"] == -1
    assert row["label_drowsiness"] == label
    assert row["metadata"]["glasses"] == 1
    assert row["metadata"]["lighting"] == 0


@pytest.mark.parametrize("name", ["open.png", "s0001_1_0_0_2_0_0_01.png"])
def test_mrl_rejects_lost_identity_or_bad_label(name):
    with pytest.raises(ValueError):
        parse_filename(name)


def test_subject_splits_persist_and_reject_leakage(tmp_path):
    rows = records()
    path = tmp_path / "splits.json"
    parts = saved_splits(rows, path)
    assert parts == saved_splits(list(reversed(rows)), path)
    assert_no_leakage(parts)
    assert set(parts) == {"train", "val", "test"}
    assert all(parts.values())
    assert_no_leakage(fast_subset(parts, 2))
    assert {r["label_drowsiness"] for r in fast_subset(parts, 2)["train"]} == {0, 1}
    contaminated = {**parts, "val": parts["val"] + [parts["train"][0]]}
    with pytest.raises(ValueError, match="Subject leakage"):
        assert_no_leakage(contaminated)
    with pytest.raises(ValueError, match="differ"):
        saved_splits(rows[:-1], path)


def test_saved_split_tampering_detected(tmp_path):
    path = tmp_path / "splits.json"
    saved_splits(records(), path)
    saved = json.loads(path.read_text())
    saved["splits"]["train"][0]["label_drowsiness"] = 3
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="contents have changed"):
        saved_splits(records(), path)


def test_insufficient_subjects_and_duplicate_samples_rejected():
    with pytest.raises(ValueError, match="three subjects"):
        make_splits(records()[:2])
    rows = records()
    with pytest.raises(ValueError, match="Duplicate sample"):
        make_splits(rows + [rows[0]])


def test_loader_common_schema_and_image_root_boundary(tmp_path):
    file = tmp_path / "s0001_00001_0_0_1_0_1_01.png"
    Image.new("L", (20, 20), 100).save(file)
    rows = index_mrl(tmp_path)
    sample = ImageDataset(tmp_path, rows, image_transform(64))[0]
    assert set(sample) == {
        "image",
        "label_distraction",
        "label_drowsiness",
        "subject_id",
        "dataset",
        "metadata",
    }
    assert sample["image"].shape == (3, 64, 64)
    with pytest.raises(ValueError, match="escapes"):
        ImageDataset(tmp_path, [{**rows[0], "path": "../outside.png"}])[0]


def test_manifest_blocks_dmd_and_missing_subject(tmp_path):
    path = tmp_path / "manifest.csv"
    for dataset, subject, error in [("dmd", "s1", "held out"), ("yawdd", "", "identity")]:
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerows(
                [
                    ["path", "subject_id", "dataset", "label_distraction", "label_drowsiness"],
                    ["frame.png", subject, dataset, -1, 2],
                ]
            )
        with pytest.raises(ValueError, match=error):
            load_manifest(path)


def test_synthetic_landmark_ratios_and_pitch():
    eye = np.array([[0, 0], [1, 1], [3, 1], [4, 0], [3, -1], [1, -1]])
    mouth = np.array([[0, 0], [1, 1], [2, 1], [3, 1], [4, 0], [3, -1], [2, -1], [1, -1]])
    assert eye_aspect_ratio(eye) == pytest.approx(0.5)
    assert mouth_aspect_ratio(mouth) == pytest.approx(0.5)
    assert eye_aspect_ratio(eye * 3 + 20) == pytest.approx(0.5)
    theta = np.deg2rad(30)
    matrix = np.eye(4)
    matrix[1:3, 1:3] = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    assert pitch_degrees(matrix) == pytest.approx(30)
    with pytest.raises(ValueError):
        eye_aspect_ratio(np.zeros((6, 2)))


def test_blink_is_not_sleep_and_long_gap_resets_persistence():
    rules = TemporalRules()
    assert rules.update(0, 0.1, 0.1, 0)["state"] == 0
    assert rules.update(0.1, 0.1, 0.1, 0)["state"] == 0
    assert rules.update(0.2, 0.3, 0.1, 0)["state"] == 0
    assert rules.update(10, 0.1, 0.1, 0)["state"] == 0
    assert rules.update(10.1)["state"] is None
    with pytest.raises(ValueError, match="increasing"):
        rules.update(10.1, 0.1, 0.1, 0)


def test_perclos_is_duration_weighted_and_missing_does_not_mean_open():
    t = replace(
        Thresholds(), window_seconds=1, max_gap_seconds=1, persist_seconds=10, minimum_coverage=0.8
    )
    rules = TemporalRules(t)
    rules.update(0, 0.1, 0.1, 0)
    rules.update(0.9, 0.3, 0.1, 0)
    result = rules.update(1, 0.3, 0.1, 0)
    assert result["perclos"] == pytest.approx(0.9)
    assert result["coverage"] == pytest.approx(1)
    assert result["state"] == 1
    assert rules.update(1.1)["state"] is None


def test_yawning_and_pitch_need_sustained_observations():
    for mar, pitch, state in [(0.8, 0, 2), (0.1, 30, 3)]:
        rules = TemporalRules(replace(Thresholds(), persist_seconds=0.4))
        assert rules.update(0, 0.3, mar, pitch)["state"] == 0
        assert rules.update(0.4, 0.3, mar, pitch)["state"] == state


def test_threshold_tuning_cannot_use_test_rows_and_missing_is_penalized():
    row = dict(
        split="test",
        subject_id="mock:p1",
        clip_id="one",
        timestamp=0,
        ear=None,
        mar=None,
        pitch=None,
        label_drowsiness=1,
    )
    with pytest.raises(ValueError, match="validation"):
        tune_thresholds([row], [Thresholds()])
    report = score([row], Thresholds())
    assert report["missing_detections"] == 1
    assert report["macro_f1"] == 0
    assert report["confusion_matrix"][1][4] == 1


def test_masked_loss_ignores_missing_labels_and_handles_all_missing():
    logits = torch.tensor([[2.0, 0.0], [99.0, -99.0]], requires_grad=True)
    labels = torch.tensor([0, -1])
    loss = masked_cross_entropy(logits, labels)
    expected = torch.nn.functional.cross_entropy(logits[:1], labels[:1])
    assert torch.allclose(loss, expected)
    loss.backward()
    assert torch.equal(logits.grad[1], torch.zeros(2))
    logits = torch.randn(2, 2, requires_grad=True)
    loss = masked_cross_entropy(logits, torch.tensor([-1, -1]))
    loss.backward()
    assert loss.item() == 0 and torch.equal(logits.grad, torch.zeros_like(logits))


def test_supported_yawning_labels_keep_project_mapping():
    assert encode_labels(torch.tensor([0, 2, -1]), [0, 2]).tolist() == [0, 1, -1]
    with pytest.raises(ValueError, match="outside"):
        encode_labels(torch.tensor([1]), [0, 2])


def test_one_cpu_training_step_changes_classifier_weights():
    seed_everything(42)
    model = build_baseline()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    before = model.classifier[-1].weight.detach().clone()
    loss = masked_cross_entropy(model(torch.randn(2, 3, 64, 64)), torch.tensor([0, 1]))
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert not torch.equal(before, model.classifier[-1].weight)


def test_metric_csv_confusion_has_one_column_per_class(tmp_path):
    report = classification_metrics([0, 1], [0, 1], ["Alert", "Closed"])
    assert report["macro_f1"] == 1
    save_metrics(report, tmp_path, "synthetic")
    with (tmp_path / "synthetic_confusion.csv").open() as stream:
        rows = list(csv.reader(stream))
    assert rows == [["true/pred", "Alert", "Closed"], ["Alert", "1", "0"], ["Closed", "0", "1"]]


def test_training_pipeline_artifacts_on_synthetic_fixture_only(tmp_path):
    # Integration verification only. These artificial pixels are not research results.
    data = tmp_path / "data"
    data.mkdir()
    for subject in range(1, 7):
        for eye in (0, 1):
            Image.new("RGB", (64, 64), (eye * 120, 40, 80)).save(
                data / f"s{subject:04d}_00001_0_0_{eye}_0_1_01.png"
            )
    config = dict(
        seed=42,
        split_fractions=[0.7, 0.15, 0.15],
        fast_samples_per_split=8,
        image_size=64,
        batch_size=4,
        num_workers=0,
        pretrained=False,
        learning_rate=0.001,
        fast_epochs=1,
        epochs=1,
    )
    output = tmp_path / "output"
    summary = train(config, data, output, tmp_path / "splits.json", fast=True)
    assert summary["fast"] is True
    assert (output / "best.pt").is_file()
    assert (output / "test.json").is_file()
    assert (output / "environment.txt").is_file()
    assert_no_leakage(json.loads((output / "used_splits.json").read_text()))
