"""Fetch checksum-verified RT-BENE files from the official Zenodo record."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

RECORD_URL = "https://zenodo.org/api/records/3685316"
ALLOWED_PREFIX = RECORD_URL + "/files/"
MAX_FILE_BYTES = 200_000_000


def fetch_file(item, output):
    name, size, checksum = item["key"], item["size"], item["checksum"]
    url = item["links"]["self"]
    if Path(name).name != name or not url.startswith(ALLOWED_PREFIX):
        raise ValueError(f"Unexpected Zenodo file metadata: {name}")
    if size > MAX_FILE_BYTES or not checksum.startswith("md5:"):
        raise ValueError(f"Unexpected Zenodo size/checksum: {name}")
    target = output / name
    if target.exists():
        if (
            target.stat().st_size == size
            and hashlib.md5(target.read_bytes()).hexdigest() == checksum[4:]
        ):
            return target
        raise ValueError(f"Existing file failed official checksum: {target}")
    temporary = output / (name + ".part")
    digest = hashlib.md5()
    count = 0
    try:
        with urllib.request.urlopen(url, timeout=90) as response, temporary.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                count += len(chunk)
                if count > size:
                    raise ValueError(f"Download exceeded official size: {name}")
                stream.write(chunk)
                digest.update(chunk)
        if count != size or digest.hexdigest() != checksum[4:]:
            raise ValueError(f"Download failed official checksum/size: {name}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def fetch(output, subjects=None):
    output = Path(output)
    with urllib.request.urlopen(RECORD_URL, timeout=30) as response:
        record = json.load(response)
    if record.get("id") != 3685316 or record["metadata"]["license"]["id"] != "cc-by-nc-sa-4.0":
        raise ValueError("Unexpected RT-BENE Zenodo record")
    files = {item["key"]: item for item in record["files"]}
    selected = [f"s{i:03d}" for i in range(17) if i != 6] if subjects is None else subjects
    if not selected or any(s not in {f"s{i:03d}" for i in range(17)} for s in selected):
        raise ValueError("Invalid subject selection")
    names = ["rt_bene_subjects.csv"]
    for subject in selected:
        if subject == "s006":
            continue  # The official subject table marks s006 discarded.
        names.extend([f"{subject}_blink_labels.csv", f"{subject}_noglasses_eyes.tar"])
    if any(name not in files for name in names):
        raise ValueError("Official record is missing expected RT-BENE files")
    output.mkdir(parents=True, exist_ok=True)
    fetched = []
    for name in names:
        path = fetch_file(files[name], output)
        fetched.append(path.name)
        print(f"verified {path.name}", flush=True)
    return fetched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+")
    args = parser.parse_args()
    print(json.dumps(fetch(args.output, args.subjects), indent=2))


if __name__ == "__main__":
    main()
