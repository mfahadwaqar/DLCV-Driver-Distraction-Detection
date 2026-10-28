"""Explicit frame manifest for requested-access datasets; never invent labels/subjects."""

import csv
import json
from pathlib import Path

from src.data.schema import validate_labels


def load_manifest(path):
    required = {"path", "subject_id", "dataset", "label_distraction", "label_drowsiness"}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not required <= set(reader.fieldnames or []):
            raise ValueError(
                "Manifest must provide paths, real subject IDs, dataset and both labels"
            )
        rows = []
        for raw in reader:
            dataset, subject = raw["dataset"].strip(), raw["subject_id"].strip()
            if not dataset or not subject:
                raise ValueError("Missing dataset/subject identity")
            if dataset.lower() == "dmd":
                raise ValueError("DMD is held out and cannot enter Phase 1 training")
            distraction, drowsiness = int(raw["label_distraction"]), int(raw["label_drowsiness"])
            validate_labels(distraction, drowsiness)
            rows.append(
                {
                    "path": raw["path"],
                    "subject_id": f"{dataset}:{subject}",
                    "dataset": dataset,
                    "label_distraction": distraction,
                    "label_drowsiness": drowsiness,
                    "metadata": json.loads(raw.get("metadata") or "{}"),
                }
            )
    if not rows:
        raise ValueError("Empty manifest")
    return rows
