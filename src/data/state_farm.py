"""Index only labeled State Farm training images with original driver identities."""

import argparse
import csv
import json
import re
from pathlib import Path

# Project IDs: safe, phone, drinking, other object, gaze (unsupported), passenger.
CLASS_MAP = {
    "c0": 0,
    "c1": 1,
    "c2": 1,
    "c3": 1,
    "c4": 1,
    "c5": 3,
    "c6": 2,
    "c7": 3,
    "c8": 3,
    "c9": 5,
}


def index_state_farm(root):
    root = Path(root).resolve()
    candidates = list(root.rglob("driver_imgs_list.csv"))
    if len(candidates) != 1:
        raise ValueError("Expected one official driver_imgs_list.csv beneath data root")
    base = candidates[0].parent
    rows, seen = [], set()
    with candidates[0].open(newline="", encoding="utf-8-sig") as stream:
        for record in csv.DictReader(stream):
            subject, source, image = record["subject"], record["classname"], record["img"]
            if not re.fullmatch(r"p[0-9]+", subject) or source not in CLASS_MAP:
                raise ValueError("Invalid State Farm driver or class")
            if not re.fullmatch(r"img_[0-9]+\.jpg", image) or image in seen:
                raise ValueError("Invalid or repeated State Farm image")
            seen.add(image)
            choices = [base / "imgs/train" / source / image, base / "train" / source / image]
            paths = [path for path in choices if path.is_file()]
            if len(paths) != 1:
                raise FileNotFoundError(f"Expected one labeled training image: {source}/{image}")
            path = paths[0].resolve()
            if not path.is_relative_to(root):
                raise ValueError("Image escapes data root")
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "subject_id": "state_farm:" + subject,
                    "dataset": "state_farm",
                    "label_distraction": CLASS_MAP[source],
                    "label_drowsiness": -1,
                    "metadata": {"source_class": source},
                }
            )
    if not rows:
        raise ValueError("No labeled training images")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = index_state_farm(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({**row, "metadata": json.dumps(row["metadata"])} for row in rows)
    print(f"Indexed {len(rows)} labeled images from {len({r['subject_id'] for r in rows})} drivers")


if __name__ == "__main__":
    main()
