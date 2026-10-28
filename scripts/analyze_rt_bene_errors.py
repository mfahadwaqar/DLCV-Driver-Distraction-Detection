"""Summarize eye-state errors by held-out subject from one Kaggle result folder."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def analyze(artifacts, output):
    report = json.loads((artifacts / "test.json").read_text(encoding="utf-8"))
    if report.get("task") != "eye-state" or report.get("dataset_scope") != ["rt_bene"]:
        raise ValueError("Expected an RT-BENE eye-state test report")
    samples = json.loads((artifacts / "used_splits.json").read_text(encoding="utf-8"))["test"]
    with (artifacts / "test_predictions.csv").open(newline="", encoding="utf-8") as stream:
        predictions = list(csv.DictReader(stream))
    if len(samples) != len(predictions) or len(samples) != report["samples"]:
        raise ValueError("Test samples and predictions have different lengths")

    by_subject = defaultdict(Counter)
    false_negatives, false_positives = [], []
    for sample, prediction in zip(samples, predictions, strict=True):
        truth = int(sample["label_drowsiness"])
        predicted = int(prediction["prediction"])
        if truth not in (0, 1) or predicted not in (0, 1):
            raise ValueError("Expected binary eye-state labels")
        if int(prediction["label"]) != truth:
            raise ValueError("Prediction order does not match the saved test split")
        subject = sample["subject_id"]
        by_subject[subject][(truth, predicted)] += 1
        if truth != predicted:
            mistake = {
                "subject_id": subject,
                "path": sample["path"],
                "true_label": truth,
                "prediction": predicted,
                "p_closed": float(prediction["p_1"]),
            }
            (false_negatives if truth == 1 else false_positives).append(mistake)

    total = Counter()
    rows = []
    for subject, counts in sorted(by_subject.items()):
        total.update(counts)
        tp, fn, fp, tn = (counts[1, 1], counts[1, 0], counts[0, 1], counts[0, 0])
        precision = ratio(tp, tp + fp)
        recall = ratio(tp, tp + fn)
        rows.append(
            {
                "subject_id": subject,
                "images": sum(counts.values()),
                "closed_images": tp + fn,
                "closed_detected": tp,
                "closed_missed": fn,
                "open_false_alarms": fp,
                "closed_precision": precision,
                "closed_recall": recall,
                "closed_f1": ratio(2 * tp, 2 * tp + fp + fn),
                "open_correct": tn,
            }
        )
    if [[total[0, 0], total[0, 1]], [total[1, 0], total[1, 1]]] != report[
        "confusion_matrix"
    ]:
        raise ValueError("Per-subject counts do not reproduce the saved confusion matrix")

    output.mkdir(parents=True, exist_ok=True)
    with (output / "by_subject.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name, mistakes in (
        ("false_negatives.csv", sorted(false_negatives, key=lambda row: row["p_closed"])),
        ("false_positives.csv", sorted(false_positives, key=lambda row: -row["p_closed"])),
    ):
        with (output / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["subject_id", "path", "true_label", "prediction", "p_closed"],
            )
            writer.writeheader()
            writer.writerows(mistakes)
    summary = {
        "test_images": len(samples),
        "subjects": rows,
        "false_negatives": len(false_negatives),
        "false_positives": len(false_positives),
        "limitations": (
            "No lighting or glasses condition label is present in these prepared samples."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.artifacts, args.output), indent=2))


if __name__ == "__main__":
    main()
