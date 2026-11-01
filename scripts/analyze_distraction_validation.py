"""Inspect State Farm validation errors only; never score or open test images."""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.data.schema import DISTRACTION
from src.data.state_farm import CLASS_MAP

CLASSES = [0, 1, 2, 3, 5]


def analyze(artifacts, image_root, output):
    report = json.loads((artifacts / "val.json").read_text(encoding="utf-8"))
    if report.get("split") != "val" or report.get("dataset_scope") != ["state_farm"]:
        raise ValueError("Expected State Farm validation metrics")
    if report["class_order"] != [DISTRACTION[i] for i in CLASSES]:
        raise ValueError("Unexpected class order")
    rows = json.loads((artifacts / "used_splits.json").read_text(encoding="utf-8"))["val"]
    with (artifacts / "val_predictions.csv").open(newline="", encoding="utf-8") as stream:
        predictions = list(csv.DictReader(stream))
    if len(rows) != len(predictions) or len(rows) != report["samples"]:
        raise ValueError("Validation sample and prediction counts differ")
    matrix = [[0] * len(CLASSES) for _ in CLASSES]
    subjects = defaultdict(Counter)
    errors = []
    for row, prediction in zip(rows, predictions, strict=True):
        truth, predicted = row["label_distraction"], int(prediction["prediction"])
        if row["dataset"] != "state_farm" or int(prediction["label"]) != truth:
            raise ValueError("Prediction order/labels differ from saved validation split")
        source = row["metadata"]["source_class"]
        if CLASS_MAP[source] != truth or predicted not in CLASSES:
            raise ValueError("Source-to-project mapping or predicted class is invalid")
        probability = float(prediction[f"p_{predicted}"])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid prediction probability")
        matrix[CLASSES.index(truth)][CLASSES.index(predicted)] += 1
        subjects[row["subject_id"]]["images"] += 1
        subjects[row["subject_id"]]["correct"] += truth == predicted
        if truth != predicted:
            errors.append(
                {
                    "path": row["path"],
                    "subject_id": row["subject_id"],
                    "source_class": source,
                    "truth": truth,
                    "prediction": predicted,
                    "confidence": probability,
                }
            )
    if matrix != report["confusion_matrix"]:
        raise ValueError("Predictions do not reproduce saved validation confusion matrix")
    output.mkdir(parents=True, exist_ok=True)
    errors.sort(key=lambda row: -row["confidence"])
    with (output / "errors.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "path",
                "subject_id",
                "source_class",
                "truth",
                "prediction",
                "confidence",
            ],
        )
        writer.writeheader()
        writer.writerows(errors)
    image_root = image_root.resolve()
    sheets = []
    for truth, predicted in [(0, 5), (5, 3), (0, 3)]:
        selected = [r for r in errors if (r["truth"], r["prediction"]) == (truth, predicted)][:12]
        if not selected:
            continue
        sheet = Image.new("RGB", (1280, math.ceil(len(selected) / 4) * 290), "#eeeeee")
        draw = ImageDraw.Draw(sheet)
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except OSError:
            font = ImageFont.load_default()
        for i, row in enumerate(selected):
            path = (image_root / row["path"]).resolve()
            if not path.is_relative_to(image_root):
                raise ValueError("Image escapes prepared root")
            x, y = i % 4 * 320, i // 4 * 290
            with Image.open(path) as opened:
                image = opened.convert("RGB")
                image.thumbnail((320, 240))
                sheet.paste(image, (x, y))
            text = (
                f"{row['subject_id'].split(':')[-1]} {row['source_class']} "
                f"{truth} -> {predicted} p={row['confidence']:.3f}\n{Path(row['path']).name}"
            )
            draw.text((x + 5, y + 242), text, font=font, fill="black")
        name = f"validation_{truth}_as_{predicted}.jpg"
        sheet.save(output / name, quality=90)
        sheets.append(name)
    summary = {
        "split": "val",
        "samples": len(rows),
        "errors": len(errors),
        "source_mapping_verified": True,
        "confusion_matrix": matrix,
        "subjects": {
            subject: {**counts, "accuracy": counts["correct"] / counts["images"]}
            for subject, counts in sorted(subjects.items())
        },
        "contact_sheets": sheets,
        "sampling": (
            "Up to 12 highest-confidence errors per selected confusion pair; not representative"
        ),
        "limitations": (
            "Filename/CSV labels verified; visual review does not independently relabel data."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.artifacts, args.image_root, args.output), indent=2))


if __name__ == "__main__":
    main()
