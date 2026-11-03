"""Prepare local VIA labels and contact sheets; never infer identities or event labels."""

import argparse
import csv
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def via_template(labels):
    """VIA 3.1.1 project data, with no files, identities or temporal annotations."""
    return {
        "project": {
            "pid": "__VIA_PROJECT_ID__",
            "rev": "__VIA_PROJECT_REV_ID__",
            "rev_timestamp": "__VIA_PROJECT_REV_TIMESTAMP__",
            "pname": "Driver monitoring development annotation",
            "data_format_version": "3.1.1",
            "creator": "VGG Image Annotator; local driver-monitoring label template",
            "created": int(datetime.now(timezone.utc).timestamp() * 1000),
            "vid_list": [],
        },
        "config": {
            "file": {"loc_prefix": {"1": "", "2": "", "3": "", "4": ""}},
            "ui": {
                "file_content_align": "center",
                "file_metadata_editor_visible": True,
                "spatial_metadata_editor_visible": False,
                "temporal_segment_metadata_editor_visible": True,
                "spatial_region_label_attribute_id": "",
                "gtimeline_visible_row_count": "5",
            },
        },
        "attribute": {
            "1": {
                "aname": "event_label",
                "anchor_id": "FILE1_Z2_XY0",
                "type": 4,
                "desc": "Reviewed temporal event. Different tasks may overlap; blank is unknown.",
                "options": {str(i): name for i, name in enumerate(labels)},
                "default_option_id": "",
            },
            "2": {
                "aname": "notes",
                "anchor_id": "FILE1_Z2_XY0",
                "type": 1,
                "desc": "Evidence, uncertainty or negative-example subtype",
                "options": {},
                "default_option_id": "",
            },
        },
        "file": {},
        "metadata": {},
        "view": {},
    }


def prepare(project, output):
    project, output = Path(project).resolve(), Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Preserving existing annotation work; choose an empty output directory")
    app = project / ".cache/annotation_tools/via_video_annotator.html"
    if not app.is_file():
        raise FileNotFoundError(
            "Download the official VIA 3 video HTML first; see annotation guide"
        )
    videos = sorted((project / "data/nitymed/source").glob("*.mp4"))
    if not videos:
        raise FileNotFoundError("No canonical NITYMED source videos found")
    labels = json.loads((project / "configs/annotation_labels.json").read_text(encoding="utf-8"))
    usage = project / "data/demo/development_usage.json"
    exposed = (
        {row["sha256"] for row in json.loads(usage.read_text(encoding="utf-8"))}
        if usage.exists()
        else set()
    )
    output.mkdir(parents=True, exist_ok=True)
    for folder in ("contact_sheets", "exports"):
        (output / folder).mkdir(exist_ok=True)
    shutil.copy2(app, output / app.name)
    (output / "via_driver_events_template.json").write_text(
        json.dumps(via_template(labels), indent=2), encoding="utf-8"
    )
    (output / "label_definitions.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    records, sheets, sheet = [], [], None
    for index, video in enumerate(videos):
        if index % 6 == 0:
            sheet = Image.new("RGB", (960, 6 * 218), "#eeeeee")
        capture = cv2.VideoCapture(str(video))
        try:
            count, fps = capture.get(cv2.CAP_PROP_FRAME_COUNT), capture.get(cv2.CAP_PROP_FPS)
            if (
                not capture.isOpened()
                or not math.isfinite(count)
                or not math.isfinite(fps)
                or count < 1
                or fps <= 0
            ):
                raise ValueError(f"Invalid video metadata: {video.name}")
            duration = count / fps
            y = (index % 6) * 218
            draw = ImageDraw.Draw(sheet)
            draw.text(
                (8, y + 3),
                f"{video.name} | {duration:.2f}s | participant: ______",
                font=font,
                fill="black",
            )
            for column, fraction in enumerate((0.1, 0.5, 0.9)):
                frame_index = min(int(count) - 1, int(count * fraction))
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"Cannot read identity-review thumbnail: {video.name}")
                thumb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                thumb.thumbnail((312, 175))
                sheet.paste(thumb, (column * 320 + 4, y + 26))
                draw.text(
                    (column * 320 + 8, y + 198),
                    f"{frame_index / fps:.2f}s",
                    font=font,
                    fill="black",
                )
        finally:
            capture.release()
        digest = sha256(video)
        records.append(
            {
                "video_file": video.name,
                "sha256": digest,
                "duration_seconds": round(duration, 6),
                "participant_id": "",
                "session_id": "",
                "identity_evidence": "",
                "identity_status": "unreviewed",
                "reviewer": "",
                "notes": "",
                "development_exposure": "recorded_demo" if digest in exposed else "not_recorded",
            }
        )
        if index % 6 == 5 or index == len(videos) - 1:
            name = f"contact_sheets/review_{index // 6 + 1:03}.jpg"
            sheet.save(output / name, quality=88)
            sheets.append(name)
        if (index + 1) % 12 == 0 or index == len(videos) - 1:
            print(f"Prepared review thumbnails: {index + 1}/{len(videos)}", flush=True)
    with (output / "identity_review.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    manifest = {
        "videos": len(records),
        "duration_seconds": sum(r["duration_seconds"] for r in records),
        "contact_sheets": sheets,
        "identities_assigned": 0,
        "event_annotations_created": 0,
        "source_videos_modified": False,
        "via_sha256": sha256(app),
        "duration_basis": "OpenCV frame count / nominal FPS; not event alignment",
        "via_source": "https://www.robots.ox.ac.uk/~vgg/software/via/app/via_video_annotator.html",
    }
    (output / "starter_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    links = "\n".join(f'<li><a href="{name}">{name}</a></li>' for name in sheets)
    (output / "START_HERE.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Driver annotation starter</title>'
        "<h1>Driver annotation starter</h1>"
        "<p>1. Review the contact sheets and fill identity_review.csv. "
        "IDs are blank deliberately.</p>"
        "<p>2. Open the annotator below. Use its Open a VIA Project button to load "
        "via_driver_events_template.json, then add local videos with the plus-circle button.</p>"
        "<p>3. Annotate only three pilot clips first. Save the project with VIA's Save button "
        "and place its JSON under exports/. Do not use the browser Save command.</p>"
        '<p><a href="via_video_annotator.html">Open local VIA video annotator</a></p>'
        '<p><a href="identity_review.csv">Driver identity review CSV</a></p>'
        "<p>Read docs/ANNOTATION_AND_FINETUNING_GUIDE.md in the project for the full protocol.</p>"
        f"<h2>Contact sheets</h2><ul>{links}</ul>",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("data/annotations/starter"))
    args = parser.parse_args()
    prepare(args.project, args.output)


if __name__ == "__main__":
    main()
