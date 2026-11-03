import json

import pytest

from scripts.build_yawn_labeler import BOOT, END, START, build


def test_preloads_only_yawn_labels_without_creating_annotations(tmp_path):
    original = f"<html><script>{BOOT}</script></html>"
    app = tmp_path / ".cache/annotation_tools/via_video_annotator.html"
    app.parent.mkdir(parents=True)
    app.write_text(original)
    output = tmp_path / "labeler.html"
    metadata = build(tmp_path, output)
    generated = output.read_text()
    assert generated.count(START) == generated.count(END) == 1
    encoded = generated.split("var driver_yawn_preset = ", 1)[1].split(";\n", 1)[0]
    preset = json.loads(encoded)
    assert list(preset["attribute"]["1"]["options"].values()) == [
        "yawn_yes",
        "yawn_no",
        "yawn_uncertain",
    ]
    assert preset["attribute"]["1"]["default_option_id"] == ""
    assert not preset["metadata"] and not preset["file"]
    assert metadata["annotations_created"] == 0
    assert app.read_text() == original
    with pytest.raises(ValueError, match="Preserving"):
        build(tmp_path, output)


def test_unknown_app_startup_is_not_rewritten(tmp_path):
    app = tmp_path / ".cache/annotation_tools/via_video_annotator.html"
    app.parent.mkdir(parents=True)
    app.write_text("<html>unknown version</html>")
    output = tmp_path / "labeler.html"
    with pytest.raises(ValueError, match="Unrecognized"):
        build(tmp_path, output)
    assert not output.exists()
