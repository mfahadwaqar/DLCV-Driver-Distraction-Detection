"""Prepare official RT-BENE subject archives for the eye-state baseline."""

import argparse
import csv
import json
import shutil
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath


def subject_table(path):
    rows = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        for record in csv.reader(stream):
            if len(record) != 6:
                raise ValueError(f"Invalid RT-BENE subject row: {record}")
            number, label_file, left, right, category, fold = record
            subject = f"s{int(number):03d}"
            if not label_file.startswith(subject + "_") or subject in rows:
                raise ValueError(f"Invalid or repeated subject: {subject}")
            if not left.startswith(subject + "_") or not right.startswith(subject + "_"):
                raise ValueError(f"Subject paths mismatch: {subject}")
            rows[subject] = {
                "labels": label_file,
                "left": left,
                "right": right,
                "category": category,
                "fold": int(fold),
            }
    if not rows:
        raise ValueError("Empty RT-BENE subject table")
    return rows


def label_table(path):
    labels = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        for record in csv.reader(stream):
            if len(record) != 2:
                raise ValueError(f"Invalid RT-BENE label row: {record}")
            name, value = record
            if PurePosixPath(name).name != name or not name.startswith("left_"):
                raise ValueError(f"Invalid labeled filename: {name}")
            if name in labels or value not in {"0.0", "0.5", "1.0"}:
                raise ValueError(f"Repeated filename or invalid label: {name}, {value}")
            labels[name] = value
    return labels


def prepare_subject(source, output, subject, info):
    archive = source / f"{subject}_noglasses_eyes.tar"
    labels = label_table(source / info["labels"])
    expected_prefix = info["left"].rstrip("/") + "/"
    found = set()
    rows = []
    with tarfile.open(archive) as contents:
        for member in contents:
            if not member.isfile() or not member.name.startswith(expected_prefix):
                continue
            name = member.name[len(expected_prefix) :]
            if name not in labels or name in found:
                raise ValueError(f"Unexpected or duplicate left-eye image: {member.name}")
            found.add(name)
            label = labels[name]
            if label == "0.5":
                continue
            target = output / subject / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with contents.extractfile(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            rows.append(
                {
                    "path": target.relative_to(output).as_posix(),
                    "subject_id": subject,
                    "dataset": "rt_bene",
                    "label_distraction": -1,
                    "label_drowsiness": 1 if label == "1.0" else 0,
                    "metadata": json.dumps(
                        {"source_fold": info["fold"], "source_category": info["category"]}
                    ),
                }
            )
    if found != set(labels):
        raise ValueError(f"Labeled images missing from {archive.name}: {len(set(labels) - found)}")
    return rows


def prepare(source, output, subjects=None):
    source, output = Path(source), Path(output)
    table = subject_table(source / "rt_bene_subjects.csv")
    selected = set(table) if subjects is None else set(subjects)
    if not selected <= set(table):
        raise ValueError(f"Unknown subjects: {sorted(selected - set(table))}")
    rows = []
    for subject in sorted(selected):
        if table[subject]["category"] == "discarded":
            continue
        rows.extend(prepare_subject(source, output, subject, table[subject]))
    if not rows:
        raise ValueError("No supervised RT-BENE images prepared")
    manifest = output / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "manifest": str(manifest),
        "subjects": sorted({row["subject_id"] for row in rows}),
        "labels": dict(Counter(row["label_drowsiness"] for row in rows)),
        "excluded_disagreements": sum(
            sum(value == "0.5" for value in label_table(source / table[s]["labels"]).values())
            for s in selected
            if table[s]["category"] != "discarded"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, args.subjects), indent=2))


if __name__ == "__main__":
    main()
