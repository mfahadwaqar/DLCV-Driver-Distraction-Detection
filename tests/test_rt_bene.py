"""RT-BENE adapter tests using the archive layout verified from a real sample."""

import csv
import io
import tarfile

import pytest

from src.data.manifest import load_manifest
from src.data.rt_bene import prepare


def test_prepare_excludes_disagreement_and_unlabeled_right_eye(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "rt_bene_subjects.csv").write_text(
        "12,s012_blink_labels.csv,s012_noglasses/natural/left/,"
        "s012_noglasses/natural/right/,training,2\n"
    )
    with (source / "s012_blink_labels.csv").open("w", newline="") as stream:
        csv.writer(stream).writerows(
            [
                ("left_000001_rgb.png", "0.0"),
                ("left_000002_rgb.png", "1.0"),
                ("left_000003_rgb.png", "0.5"),
            ]
        )
    with tarfile.open(source / "s012_noglasses_eyes.tar", "w") as archive:
        for side in ("left", "right"):
            for number in (1, 2, 3):
                name = f"s012_noglasses/natural/{side}/{side}_{number:06d}_rgb.png"
                payload = b"sample"
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
    output = tmp_path / "prepared"
    summary = prepare(source, output)
    rows = load_manifest(output / "manifest.csv")
    assert summary["labels"] == {0: 1, 1: 1}
    assert summary["excluded_disagreements"] == 1
    assert {row["subject_id"] for row in rows} == {"rt_bene:s012"}
    assert all(row["path"].startswith("s012/left_") for row in rows)
    assert len(list(output.rglob("*.png"))) == 2


def test_prepare_rejects_missing_labeled_image(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "rt_bene_subjects.csv").write_text(
        "12,s012_blink_labels.csv,s012_noglasses/natural/left/,"
        "s012_noglasses/natural/right/,training,2\n"
    )
    (source / "s012_blink_labels.csv").write_text("left_000001_rgb.png,0.0\n")
    with tarfile.open(source / "s012_noglasses_eyes.tar", "w"):
        pass
    with pytest.raises(ValueError, match="missing"):
        prepare(source, tmp_path / "prepared")
