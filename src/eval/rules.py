"""Tune B1 on validation subjects only, then score once on held-out test features."""

import argparse
import csv
import json
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path

from src.data.schema import DROWSINESS
from src.data.splits import assert_no_leakage
from src.eval.metrics import save_metrics
from src.models.rules import TemporalRules, Thresholds


def read_features(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {
            "subject_id",
            "clip_id",
            "timestamp",
            "ear",
            "mar",
            "pitch",
            "label_drowsiness",
            "split",
        }
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Feature CSV is missing required columns")
        rows = []
        for raw in reader:
            row = {
                **raw,
                "timestamp": float(raw["timestamp"]),
                "label_drowsiness": int(raw["label_drowsiness"]),
            }
            if row["label_drowsiness"] not in range(-1, 4):
                raise ValueError("Invalid drowsiness label")
            for name in ("ear", "mar", "pitch"):
                row[name] = float(raw[name]) if raw[name] else None
            rows.append(row)
    parts = {name: [] for name in ("train", "val", "test")}
    seen = set()
    for row in rows:
        if row["split"] not in parts:
            raise ValueError("Assign saved subject-wise train/val/test splits before evaluation")
        key = (row["subject_id"], row["clip_id"], row["timestamp"])
        if key in seen:
            raise ValueError("Duplicate frame timestamp")
        seen.add(key)
        parts[row["split"]].append(
            {
                "subject_id": row["subject_id"],
                "dataset": "features",
                "path": f"{row['subject_id']}/{row['clip_id']}/{row['timestamp']}",
            }
        )
    assert_no_leakage(parts)
    return rows


def score(rows, thresholds):
    ordered = sorted(rows, key=lambda r: (r["subject_id"], r["clip_id"], r["timestamp"]))
    current, rules = None, None
    truth, predictions = [], []
    missing, labelled = 0, 0
    for row in ordered:
        key = row["subject_id"], row["clip_id"]
        if key != current:
            rules, current = TemporalRules(thresholds), key
        result = rules.update(row["timestamp"], row["ear"], row["mar"], row["pitch"])
        label = row["label_drowsiness"]
        if label == -1:
            continue
        labelled += 1
        truth.append(label)
        if result["state"] is None:
            # Count abstentions as errors via an explicit no-detection prediction class.
            missing += 1
            predictions.append(4)
        else:
            predictions.append(result["state"])
    if not labelled:
        raise ValueError("No annotated frames to evaluate")
    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    report = classification_report(
        truth,
        predictions,
        labels=list(range(5)),
        target_names=[*DROWSINESS, "No detection"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "macro_f1": float(
            f1_score(truth, predictions, labels=list(range(4)), average="macro", zero_division=0)
        ),
        "per_class": report,
        "class_order": [*DROWSINESS, "No detection"],
        "confusion_matrix": confusion_matrix(truth, predictions, labels=list(range(5))).tolist(),
        "samples": labelled,
        "missing_detections": missing,
        "coverage": 1 - missing / labelled,
    }


def tune_thresholds(validation, candidates):
    if not validation or any(row["split"] != "val" for row in validation):
        raise ValueError("Threshold selection accepts validation rows only")
    if not candidates:
        raise ValueError("At least one candidate required")
    return max(candidates, key=lambda candidate: score(validation, candidate)["macro_f1"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use an empty output directory")
    rows = read_features(args.features)
    candidates = [
        replace(Thresholds(), ear=e, mar=m, pitch=p, perclos=c)
        for e, m, p, c in product((0.18, 0.22), (0.4, 0.6), (15, 25), (0.3, 0.5))
    ]
    best = tune_thresholds([r for r in rows if r["split"] == "val"], candidates)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "thresholds.json").write_text(json.dumps(asdict(best), indent=2))
    for split in ("val", "test"):
        report = score([r for r in rows if r["split"] == split], best)
        report.update({"split": split, "baseline": "B1", "thresholds_selected_on": "val"})
        save_metrics(report, args.output, split)
        print(split, report["macro_f1"], "coverage", report["coverage"])


if __name__ == "__main__":
    main()
