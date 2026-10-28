"""Frame-level reports. Event metrics require annotated video and are not inferred."""

import csv
import json
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix


def classification_metrics(truth, predictions, names):
    if not truth or len(truth) != len(predictions):
        raise ValueError("Nonempty paired labels/predictions required")
    labels = list(range(len(names)))
    if not set(truth + predictions) <= set(labels):
        raise ValueError("Out-of-taxonomy predictions/labels")
    report = classification_report(
        truth, predictions, labels=labels, target_names=names, output_dict=True, zero_division=0
    )
    return {
        "macro_f1": report["macro avg"]["f1-score"],
        "per_class": report,
        "confusion_matrix": confusion_matrix(truth, predictions, labels=labels).tolist(),
        "class_order": list(names),
        "samples": len(truth),
    }


def save_metrics(report, output, stem):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{stem}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output / f"{stem}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        for name in report["class_order"]:
            row = report["per_class"][name]
            writer.writerow(
                [name, row["precision"], row["recall"], row["f1-score"], row["support"]]
            )
    with (output / f"{stem}_confusion.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true/pred", *report["class_order"]])
        writer.writerows(
            [name, *row]
            for name, row in zip(report["class_order"], report["confusion_matrix"], strict=True)
        )
