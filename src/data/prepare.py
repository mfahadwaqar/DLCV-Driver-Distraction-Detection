"""Index acquired MRL images and persist deterministic subject-disjoint manifests."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from src.data.mrl import index_mrl
from src.data.splits import saved_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = index_mrl(args.data_root)
    parts = saved_splits(rows, args.splits, args.seed)
    stats = {
        name: {
            "samples": len(group),
            "subjects": sorted({r["subject_id"] for r in group}),
            "labels": dict(Counter(r["label_drowsiness"] for r in group)),
            "glasses": sum(r["metadata"]["glasses"] == 1 for r in group),
            "bad_lighting": sum(r["metadata"]["lighting"] == 0 for r in group),
        }
        for name, group in parts.items()
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "data_summary.json").write_text(json.dumps(stats, indent=2))
    with (args.output / "data_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "subjects", "samples", "glasses", "bad_lighting"])
        for name, summary in stats.items():
            writer.writerow(
                [
                    name,
                    len(summary["subjects"]),
                    summary["samples"],
                    summary["glasses"],
                    summary["bad_lighting"],
                ]
            )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
