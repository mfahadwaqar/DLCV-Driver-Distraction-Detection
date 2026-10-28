"""CPU integration smoke run. This does not train or evaluate a model."""

import argparse
import csv
import importlib.metadata
import json
import platform
import random
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/artifacts"))
    parser.add_argument("--input", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[2] / "SOURCE_COMMIT"
    report = {
        "kind": "integration-smoke-only",
        "training_performed": False,
        "source_commit": source.read_text().strip() if source.exists() else "local-unbundled",
        "seed": args.seed,
        "fast": args.fast,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    # Bounded-depth lookup avoids traversing the image tree on the Kaggle mount.
    candidates = list(args.input.glob("*/driver_imgs_list.csv"))
    candidates += list(args.input.glob("competitions/*/driver_imgs_list.csv"))
    if candidates:
        with candidates[0].open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        report["state_farm"] = {
            "rows": len(rows),
            "subjects": len({r["subject"] for r in rows}),
            "columns": list(rows[0]) if rows else [],
        }
    else:
        report["state_farm"] = {"available": False}
    (args.output / "smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "config.json").write_text(
        json.dumps({"seed": args.seed, "fast": args.fast}, indent=2), encoding="utf-8"
    )
    versions = sorted(
        f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()
    )
    (args.output / "environment.txt").write_text("\n".join(versions) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
