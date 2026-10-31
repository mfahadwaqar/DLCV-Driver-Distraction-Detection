import json
import zipfile

import pytest

from scripts.prepare_local_data import (
    extract_state_farm,
    remap_historical,
    verify_official_metadata,
)


def row(subject="p001", path="imgs/train/c0/img_1.jpg"):
    return {
        "dataset": "state_farm",
        "subject_id": "state_farm:" + subject,
        "path": path,
        "label_distraction": 0,
        "label_drowsiness": -1,
        "metadata": {},
    }


def test_historical_membership_survives_path_and_namespace_normalization(tmp_path):
    local = [row(f"p00{i}", f"imgs/train/c0/img_{i}.jpg") for i in range(1, 4)]
    parts = {}
    for name, current in zip(("train", "val", "test"), local, strict=True):
        old = {
            **current,
            "subject_id": "state_farm:" + current["subject_id"],
            "path": "old/kaggle/" + current["path"],
        }
        parts[name] = [old]
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"seed": 42, "fractions": [0.7, 0.15, 0.15], "splits": parts}))
    saved = remap_historical(local, prior)
    assert {name: group[0]["subject_id"] for name, group in saved["splits"].items()} == {
        "train": "state_farm:p001",
        "val": "state_farm:p002",
        "test": "state_farm:p003",
    }
    local[0]["label_distraction"] = 1
    with pytest.raises(ValueError, match="samples/labels differ"):
        remap_historical(local, prior)


def test_unknown_subject_cannot_receive_new_assignment(tmp_path):
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"train": [row()], "val": [], "test": []}))
    with pytest.raises(ValueError, match="No historical assignment"):
        remap_historical([row("p099")], prior)


def test_selective_zip_extracts_only_labeled_train(tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as contents:
        contents.writestr("driver_imgs_list.csv", "subject,classname,img\np001,c0,img_1.jpg\n")
        contents.writestr("imgs/train/c0/img_1.jpg", b"train-image")
        contents.writestr("imgs/test/img_9.jpg", b"excluded-test")
        contents.writestr("../escape.jpg", b"excluded-unsafe")
        contents.writestr("sample_submission.csv", b"excluded")
    output = tmp_path / "prepared"
    report = extract_state_farm(archive, output)
    assert report["zip_members_crc_checked"] == 5
    assert report["selected_members"] == 2
    assert (output / "imgs/train/c0/img_1.jpg").read_bytes() == b"train-image"
    assert not (output / "imgs/test").exists()
    assert not (tmp_path / "escape.jpg").exists()
    assert extract_state_farm(archive, output)["selected_members"] == 2
    (output / "imgs/train/c0/img_1.jpg").write_bytes(b"earlier-artifact")
    with pytest.raises(ValueError, match="Existing prepared file differs"):
        extract_state_farm(archive, output)
    assert (output / "imgs/train/c0/img_1.jpg").read_bytes() == b"earlier-artifact"


def test_official_metadata_rejects_corrupt_file_before_extracting(tmp_path):
    (tmp_path / "s000_noglasses_eyes.tar").write_bytes(b"corrupt")
    (tmp_path / "zenodo_record.json").write_text(
        json.dumps(
            {
                "id": 3685316,
                "files": [
                    {"key": "s000_noglasses_eyes.tar", "size": 7, "checksum": "md5:" + "0" * 32}
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="Official checksum/size failed"):
        verify_official_metadata(tmp_path)


def test_persisted_splits_match_training_manifest_fingerprint(tmp_path):
    import csv

    from src.data.manifest import load_manifest
    from src.data.splits import saved_splits

    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row()))
        writer.writeheader()
        for i in range(1, 4):
            sample = row(f"p00{i}", f"imgs/train/c0/img_{i}.jpg")
            writer.writerow({**sample, "subject_id": f"p00{i}", "metadata": "{}"})
    loaded = load_manifest(manifest)
    prior = tmp_path / "historical.json"
    prior.write_text(
        json.dumps(
            dict(zip(("train", "val", "test"), [[sample] for sample in loaded], strict=True))
        )
    )
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps(remap_historical(loaded, prior)))
    actual = saved_splits(loaded, splits, seed=42, fractions=[0.7, 0.15, 0.15])
    assert actual["test"][0]["subject_id"] == "state_farm:p003"
