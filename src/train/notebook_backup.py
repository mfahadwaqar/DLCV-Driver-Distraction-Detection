"""Carry a small experiment ZIP in saved notebook output, separate from readable code."""

import argparse
import base64
import hashlib
import io
import json
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

MIME = "application/vnd.dlcv.experiment+json"
MAX_ARCHIVE = 50 * 1024**2
MAX_EXPANDED = 512 * 1024**2


def backup_payload(experiment):
    experiment = Path(experiment).resolve()
    if not (experiment / "model/best.pt").is_file():
        raise FileNotFoundError("No trained checkpoint found to back up")
    archive = Path(
        shutil.make_archive(
            str(experiment.parent / f"{experiment.name}_backup"),
            "zip",
            root_dir=experiment.parent,
            base_dir=experiment.name,
        )
    )
    if archive.stat().st_size > MAX_ARCHIVE:
        raise ValueError("Archive exceeds notebook backup limit; download ZIP separately")
    content = archive.read_bytes()
    return {
        "experiment": experiment.name,
        "sha256": hashlib.sha256(content).hexdigest(),
        "archive_base64": base64.b64encode(content).decode("ascii"),
    }


def restore_notebook(notebook, destination):
    notebook = json.loads(Path(notebook).read_text(encoding="utf-8"))
    destination = Path(destination).resolve()
    payloads = {}
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            if MIME in output.get("data", {}):
                payload = output["data"][MIME]
                payloads[payload["sha256"]] = payload
    if not payloads:
        raise ValueError("No embedded experiment backup found in saved notebook outputs")
    restored = []
    for digest, payload in payloads.items():
        name = payload["experiment"]
        if not name or name in {".", ".."} or any(c in name for c in "/\\:"):
            raise ValueError("Unsafe experiment directory")
        encoded = payload["archive_base64"]
        if len(encoded) > 4 * ((MAX_ARCHIVE + 2) // 3):
            raise ValueError("Embedded archive exceeds size limit")
        content = base64.b64decode(encoded, validate=True)
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("Backup SHA-256 mismatch")
        target = destination / name
        if target.exists():
            raise FileExistsError(f"Preserving existing experiment: {target}")
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            members = archive.infolist()
            if len(members) > 1000 or sum(m.file_size for m in members) > MAX_EXPANDED:
                raise ValueError("Expanded experiment exceeds backup limits")
            seen = set()
            for member in members:
                path = PurePosixPath(member.filename)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or "\\" in member.filename
                    or any(":" in part for part in path.parts)
                    or not path.parts
                    or path.parts[0] != name
                    or member.filename in seen
                    or stat.S_ISLNK(member.external_attr >> 16)
                ):
                    raise ValueError("Unsafe or duplicate backup member")
                seen.add(member.filename)
            if f"{name}/model/best.pt" not in seen:
                raise ValueError("Backup lacks the model checkpoint")
            if archive.testzip() is not None:
                raise ValueError("Backup ZIP integrity failed")
            with tempfile.TemporaryDirectory(dir=destination, prefix=".restore-") as temporary:
                stage = Path(temporary)
                for member in members:
                    path = stage / member.filename
                    if member.is_dir():
                        path.mkdir(parents=True, exist_ok=True)
                    else:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(member) as source, path.open("xb") as output:
                            shutil.copyfileobj(source, output)
                if target.exists():
                    raise FileExistsError(f"Preserving existing experiment: {target}")
                (stage / name).rename(target)
        restored.append(target)
    return restored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/experiments"))
    args = parser.parse_args()
    for folder in restore_notebook(args.notebook, args.output):
        print("Restored:", folder)


if __name__ == "__main__":
    main()
