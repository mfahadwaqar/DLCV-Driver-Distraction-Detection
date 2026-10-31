"""Prepare only local State Farm and RT-BENE; retain historical subject assignments.

Run: python scripts/prepare_local_data.py --datasets state_farm rt_bene
No downloads or training. All generated data and reports live beneath data/.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import zipfile
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.manifest import load_manifest
from src.data.rt_bene import label_table, prepare_subject, subject_table
from src.data.splits import NAMES, assert_no_leakage, fingerprint
from src.data.state_farm import index_state_farm

PRIOR = {
    "state_farm": "results/kaggle/20261029T065436396085Z/artifacts/state_farm_splits.json",
    "rt_bene": "results/kaggle/20261028T075933392791Z/artifacts/phase1_splits.json",
}


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def publish_file(source, target):
    """Keep existing identical files; refuse changes to existing image bytes."""
    target = Path(target)
    if target.exists():
        if digest(source) != digest(target):
            raise ValueError(f"Existing prepared file differs: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    # These sources are disposable staged copies, never the original archives.
    # A hard link avoids writing every staged image twice on the same volume.
    try:
        os.link(source, target)
        return
    except OSError:
        pass  # Other filesystems may not support links; copy with exclusive creation.
    with Path(source).open("rb") as src, target.open("xb") as dst:
        shutil.copyfileobj(src, dst)


def write_artifact(path, content):
    path = Path(path)
    if path.exists():
        previous = path.read_bytes()
        if previous == content:
            return
        backup = path.with_name(
            path.name + ".previous." + hashlib.sha256(previous).hexdigest()[:12]
        )
        if not backup.exists():
            backup.write_bytes(previous)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def canonical_subject(dataset, subject):
    prefix = dataset + ":"
    while subject.startswith(prefix):
        subject = subject[len(prefix) :]
    if ":" in subject or not subject:
        raise ValueError(f"Invalid subject: {subject}")
    return prefix + subject


def remap_historical(rows, prior):
    """Require exact sample identities and labels, allowing only path/namespace changes."""
    saved = json.loads(Path(prior).read_text(encoding="utf-8"))
    old = saved.get("splits", saved)
    assignment, samples = {}, {}
    for split in NAMES:
        for row in old[split]:
            subject = canonical_subject(row["dataset"], row["subject_id"])
            if subject in assignment and assignment[subject] != split:
                raise ValueError(f"Historical subject leakage: {subject}")
            assignment[subject] = split
            key = (row["dataset"], subject, PurePosixPath(row["path"]).name)
            if key in samples:
                raise ValueError(f"Duplicate historical sample: {key}")
            samples[key] = (row["label_distraction"], row["label_drowsiness"])
    current = {}
    parts = {name: [] for name in NAMES}
    for row in rows:
        key = (row["dataset"], row["subject_id"], PurePosixPath(row["path"]).name)
        if key in current:
            raise ValueError(f"Duplicate local sample: {key}")
        current[key] = (row["label_distraction"], row["label_drowsiness"])
        if row["subject_id"] not in assignment:
            raise ValueError(f"No historical assignment: {row['subject_id']}")
        parts[assignment[row["subject_id"]]].append(row)
    if samples != current:
        raise ValueError("Local samples/labels differ from historical split; refusing reassignment")
    for group in parts.values():
        group.sort(key=lambda r: (r["dataset"], r["path"]))
    assert_no_leakage(parts)
    return {
        "seed": saved.get("seed", 42),
        "fractions": saved.get("fractions", [0.7, 0.15, 0.15]),
        "fingerprint": fingerprint(rows),
        "splits": parts,
        "provenance": {
            "historical_split": Path(prior).as_posix(),
            "historical_sha256": digest(prior),
            "historical_fingerprint": saved.get("fingerprint"),
            "transformation": "Remap image paths; canonicalize repeated dataset namespace only.",
            "exact_sample_labels_and_subject_memberships_preserved": True,
        },
    }


def check_space(root, required):
    free = shutil.disk_usage(root).free
    if free < required + 1024**3:
        raise ValueError(f"Insufficient disk: need {required + 1024**3}, free {free}")
    return free


def verify_official_metadata(source):
    """Verify cached official metadata offline before reading any RT-BENE payload."""
    metadata = source / "zenodo_record.json"
    if not metadata.is_file():
        raise FileNotFoundError("Cached official zenodo_record.json required for checksums")
    record = json.loads(metadata.read_text(encoding="utf-8"))
    if record.get("id") != 3685316:
        raise ValueError("Unexpected RT-BENE official record")
    files = {item["key"]: item for item in record["files"]}
    verified = []
    for path in sorted(source.iterdir()):
        if path.suffix not in {".csv", ".tar"}:
            continue
        item = files[path.name]
        checksum = item["checksum"]
        if (
            not checksum.startswith("md5:")
            or path.stat().st_size != item["size"]
            or digest(path, "md5") != checksum[4:]
        ):
            raise ValueError(f"Official checksum/size failed: {path.name}")
        verified.append(path.name)
    if len(verified) != 33:
        raise ValueError("Expected official subject table and 16 archive/label pairs")
    return {
        "record_id": 3685316,
        "metadata_sha256": digest(metadata),
        "official_md5_verified_files": verified,
    }


def matches_zip_member(path, info):
    if not path.is_file() or path.stat().st_size != info.file_size:
        return False
    checksum = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum = zlib.crc32(chunk, checksum)
    return checksum == info.CRC


def extract_state_farm(archive, output):
    with zipfile.ZipFile(archive) as contents:
        selected = []
        seen = set()
        for info in contents.infolist():
            path = PurePosixPath(info.filename)
            if info.filename == "driver_imgs_list.csv" or (
                len(path.parts) == 4
                and path.parts[:2] == ("imgs", "train")
                and path.parts[2] in {f"c{i}" for i in range(10)}
                and path.name.startswith("img_")
                and path.suffix == ".jpg"
            ):
                if info.filename in seen or info.is_dir():
                    raise ValueError(f"Duplicate/invalid selected ZIP member: {info.filename}")
                seen.add(info.filename)
                selected.append(info)
        if "driver_imgs_list.csv" not in seen or len(selected) < 2:
            raise ValueError("ZIP lacks labeled State Farm training data")
        free = check_space(output.parent, 2 * sum(info.file_size for info in selected))
        print("State Farm: checking CRC of every ZIP member...", flush=True)
        bad = contents.testzip()
        if bad:
            raise ValueError(f"ZIP CRC failed: {bad}")
        print("State Farm: ZIP CRC passed; checking prepared files...", flush=True)
        complete = all(matches_zip_member(output / info.filename, info) for info in selected)
        if not complete:
            with tempfile.TemporaryDirectory(
                dir=output.parent, prefix=".state_farm_stage_"
            ) as temporary:
                stage = Path(temporary)
                for info in selected:
                    target = stage / info.filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with contents.open(info) as src, target.open("xb") as dst:
                        shutil.copyfileobj(src, dst)
                index_state_farm(stage)  # CSV must resolve every labeled train image.

                def publish_one(info):
                    publish_file(stage / info.filename, output / info.filename)

                with ThreadPoolExecutor(max_workers=8) as pool:
                    list(pool.map(publish_one, selected))
    return {
        "archive": str(archive),
        "sha256": digest(archive),
        "zip_members_crc_checked": len(contents.infolist()),
        "selected_members": len(selected),
        "free_bytes_before": free,
        "official_checksum": "Not available locally",
    }


def verify_images(rows, output):
    from PIL import Image

    output = output.resolve()

    def verify_one(row):
        path = (output / row["path"]).resolve()
        if not path.is_relative_to(output):
            raise ValueError(f"Image escapes prepared root: {path}")
        with Image.open(path) as image:
            image.load()

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(verify_one, rows))
    return len(rows)


def run(root, dataset):
    base = root / "data" / dataset
    source, output = base / "source", base / "prepared"
    base.mkdir(parents=True, exist_ok=True)
    prior = root / PRIOR[dataset]
    if not prior.is_file():
        raise FileNotFoundError(f"Historical split required: {prior}")
    if dataset == "state_farm":
        integrity = extract_state_farm(
            source / "state-farm-distracted-driver-detection.zip", output
        )
        print("State Farm: indexing original driver CSV...", flush=True)
        rows = index_state_farm(output)
        for row in rows:
            row["subject_id"] = row["subject_id"].split(":")[-1]
        excluded = 0
    else:
        print("RT-BENE: verifying official checksums and TAR payloads...", flush=True)
        official = verify_official_metadata(source)
        archives = sorted(source.glob("*.tar"))
        free = check_space(base, sum(p.stat().st_size for p in archives))
        integrity = {"free_bytes_before": free, "files": [], "official_checksum": official}
        for path in sorted(source.iterdir()):
            if path.suffix not in {".tar", ".csv"}:
                continue
            if path.suffix == ".tar":
                with tarfile.open(path) as contents:
                    for member in contents:
                        if member.isfile():
                            with contents.extractfile(member) as stream:
                                while stream.read(1024 * 1024):
                                    pass
            integrity["files"].append(
                {"name": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}
            )
        with tempfile.TemporaryDirectory(dir=base, prefix=".rt_bene_stage_") as temporary:
            stage = Path(temporary)
            table = subject_table(source / "rt_bene_subjects.csv")
            subjects = sorted(s for s, info in table.items() if info["category"] != "discarded")

            def prepare_one(subject):
                return prepare_subject(source, stage, subject, table[subject])

            # Independent subject archives can be read concurrently; ordering stays fixed.
            with ThreadPoolExecutor(max_workers=4) as pool:
                rows = [row for group in pool.map(prepare_one, subjects) for row in group]
            print(f"RT-BENE: publishing {len(rows)} supervised left-eye images...", flush=True)

            def publish_one(row):
                publish_file(stage / row["path"], output / row["path"])

            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(publish_one, rows))
            excluded = sum(
                sum(value == "0.5" for value in label_table(source / table[s]["labels"]).values())
                for s in subjects
            )
    import io

    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                **row,
                "metadata": json.dumps(row["metadata"])
                if isinstance(row["metadata"], dict)
                else row["metadata"],
            }
        )
    manifest = output / "manifest.csv"
    write_artifact(manifest, csv_buffer.getvalue().encode("utf-8"))
    loaded = load_manifest(manifest)
    print(f"{dataset}: decoding {len(loaded)} images...", flush=True)
    checked = verify_images(loaded, output)
    print(f"{dataset}: checking exact historical sample/label/split membership...", flush=True)
    saved = remap_historical(loaded, prior)
    splits = root / "data" / "splits" / f"{dataset}_splits.json"
    if splits.exists():
        existing = json.loads(splits.read_text())
        if existing["fingerprint"] != saved["fingerprint"] or existing["splits"] != saved["splits"]:
            raise ValueError(f"Existing splits differ; refusing overwrite: {splits}")
    write_artifact(splits, json.dumps(saved, indent=2).encode())
    report = {
        "dataset": dataset,
        "manifest": str(manifest),
        "splits": str(splits),
        "images_decoded": checked,
        "missing_images": 0,
        "corrupt_images": 0,
        "excluded_disagreements": excluded,
        "integrity": integrity,
        "split_provenance": saved["provenance"],
        "splits_summary": {
            name: {
                "samples": len(group),
                "subjects": sorted({r["subject_id"] for r in group}),
                "labels": dict(
                    Counter(
                        r["label_distraction" if dataset == "state_farm" else "label_drowsiness"]
                        for r in group
                    )
                ),
            }
            for name, group in saved["splits"].items()
        },
    }
    write_artifact(base / "preparation_report.json", json.dumps(report, indent=2).encode())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--datasets", nargs="+", choices=list(PRIOR), default=list(PRIOR))
    args = parser.parse_args()
    for dataset in args.datasets:
        print(json.dumps(run(args.root.resolve(), dataset), indent=2), flush=True)


if __name__ == "__main__":
    main()
