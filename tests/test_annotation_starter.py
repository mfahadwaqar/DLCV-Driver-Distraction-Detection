import csv
import json

import cv2
import numpy as np
import pytest

from scripts.prepare_annotation_starter import prepare, sha256, via_template


def test_via_template_has_no_invented_labels_identities_or_files():
    template = via_template({"yawn_yes": "Visible yawn", "yawn_no": "Reviewed negative"})
    assert template["project"]["data_format_version"] == "3.1.1"
    assert template["metadata"] == template["file"] == template["view"] == {}
    attr = template["attribute"]["1"]
    assert attr["anchor_id"] == "FILE1_Z2_XY0"
    assert attr["default_option_id"] == ""
    assert set(attr["options"].values()) == {"yawn_yes", "yawn_no"}


def test_prepare_starter_preserves_sources_and_blank_identity_fields(tmp_path):
    root = tmp_path
    (root / "configs").mkdir()
    labels = {"yawn_yes": "Visible yawn", "yawn_no": "Reviewed negative"}
    (root / "configs/annotation_labels.json").write_text(json.dumps(labels))
    app = root / ".cache/annotation_tools/via_video_annotator.html"
    app.parent.mkdir(parents=True)
    app.write_text("<!doctype html><html>fixture</html>")
    source = root / "data/nitymed/source"
    source.mkdir(parents=True)
    for index in range(3):
        writer = cv2.VideoWriter(
            str(source / f"clip{index}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 64)
        )
        assert writer.isOpened()
        for frame in range(20):
            writer.write(np.full((64, 64, 3), index * 40 + frame, dtype=np.uint8))
        writer.release()
    before = {p.name: sha256(p) for p in source.iterdir()}
    (root / "data/demo").mkdir()
    (root / "data/demo/development_usage.json").write_text(
        json.dumps([{"sha256": before["clip0.mp4"]}])
    )
    output = root / "data/annotations/starter"
    manifest = prepare(root, output)
    assert manifest["videos"] == 3 and manifest["identities_assigned"] == 0
    assert manifest["event_annotations_created"] == 0
    assert len(manifest["contact_sheets"]) == 1
    assert (output / manifest["contact_sheets"][0]).stat().st_size > 0
    assert (output / "START_HERE.html").is_file()
    assert (output / "exports").is_dir()
    with (output / "identity_review.csv").open(newline="") as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == 3
    assert all(not row["participant_id"] and not row["session_id"] for row in records)
    assert all(row["identity_status"] == "unreviewed" for row in records)
    assert records[0]["development_exposure"] == "recorded_demo"
    assert records[1]["development_exposure"] == "not_recorded"
    assert before == {p.name: sha256(p) for p in source.iterdir()}
    assert sha256(app) == sha256(output / app.name)
    with pytest.raises(ValueError, match="Preserving existing annotation work"):
        prepare(root, output)


def test_missing_tool_is_actionable_without_creating_output(tmp_path):
    output = tmp_path / "starter"
    with pytest.raises(FileNotFoundError, match="official VIA"):
        prepare(tmp_path, output)
    assert not output.exists()
