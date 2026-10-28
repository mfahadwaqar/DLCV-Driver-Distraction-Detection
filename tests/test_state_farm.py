import csv
import json

import pytest
from PIL import Image

from src.data.state_farm import CLASS_MAP, index_state_farm
from src.train.phase1 import train


def test_original_mapping_driver_splits_and_distraction_training(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    records = []
    for driver in range(6):
        for category in range(10):
            name = f"img_{driver * 10 + category}.jpg"
            path = data / "imgs/train" / f"c{category}" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 64), (category * 20, 40, 80)).save(path)
            records.append([f"p{driver:03d}", f"c{category}", name])
    with (data / "driver_imgs_list.csv").open("w", newline="") as stream:
        csv.writer(stream).writerows([["subject", "classname", "img"], *records])
    rows = index_state_farm(data)
    assert [row["label_distraction"] for row in rows[:10]] == [0, 1, 1, 1, 1, 3, 2, 3, 3, 5]
    assert all(row["label_drowsiness"] == -1 for row in rows)
    assert 4 not in CLASS_MAP.values()
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({**row, "metadata": json.dumps(row["metadata"])} for row in rows)
    config = dict(
        seed=42,
        split_fractions=[0.7, 0.15, 0.15],
        fast_samples_per_split=10,
        image_size=64,
        batch_size=5,
        num_workers=0,
        pretrained=False,
        learning_rate=0.001,
        fast_epochs=1,
        epochs=1,
        label_key="label_distraction",
        class_ids=[0, 1, 2, 3, 5],
    )
    output = tmp_path / "output"
    summary = train(config, data, output, tmp_path / "splits.json", manifest, fast=True)
    assert summary["unsupported_classes"] == ["Gaze off road"]
    report = json.loads((output / "test.json").read_text())
    assert report["task"] == "distraction"
    assert len(report["class_order"]) == 5
    records[0][1] = "c10"
    with (data / "driver_imgs_list.csv").open("w", newline="") as stream:
        csv.writer(stream).writerows([["subject", "classname", "img"], *records])
    with pytest.raises(ValueError, match="class"):
        index_state_farm(data)
