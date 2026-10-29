"""Package committed code, submit Kaggle runs, and fetch saved outputs."""

import argparse
import base64
import io
import json
import shutil
import subprocess
import sys
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / ".cache" / "kaggle-submit"
BASELINE_SPLITS = ROOT / "results/kaggle/20261029T065436396085Z/artifacts/state_farm_splits.json"


def load_baseline_splits(path=None):
    """Load only the expected public State Farm split JSON, never arbitrary files."""
    path = BASELINE_SPLITS if path is None else path
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(document) != {"seed", "fractions", "fingerprint", "splits"}:
        raise ValueError("Unexpected baseline split document fields")
    if document["seed"] != 42 or document["fractions"] != [0.7, 0.15, 0.15]:
        raise ValueError("Baseline split parameters do not match the saved experiment")
    if set(document["splits"]) != {"train", "val", "test"}:
        raise ValueError("Baseline splits must contain train, val, and test")
    for rows in document["splits"].values():
        if not isinstance(rows, list):
            raise ValueError("Each baseline split must be a list")
        for row in rows:
            if set(row) != {
                "path", "subject_id", "dataset", "label_distraction",
                "label_drowsiness", "metadata",
            }:
                raise ValueError("Unexpected fields in baseline split row")
            if not row["path"].startswith(
                "state-farm-distracted-driver-detection/imgs/train/"
            ):
                raise ValueError("Baseline split contains a path outside State Farm")
            if row["dataset"] != "state_farm" or set(row["metadata"]) != {"source_class"}:
                raise ValueError("Unexpected baseline split metadata")
    return document


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def prepare(stage=STAGE, task="smoke", data_root=None, fast=False, device="cpu"):
    if task == "phase1" and not data_root:
        raise ValueError("Phase 1 requires --data-root pointing to an attached Kaggle dataset")
    if git("status", "--porcelain", "--", "src", "configs", "scripts", "kaggle").strip():
        raise RuntimeError("Commit changes first: runs must use a clean committed snapshot.")
    commit = git("rev-parse", "HEAD").decode().strip()
    # Explicit allowlist: never bundle data, credentials, results, or .git.
    archive_paths = ["src", "configs"]
    if task == "state-farm-pretrained":
        archive_paths = ["src", "configs/state_farm_pretrained.yaml"]
    archive = git("archive", "--format=zip", "HEAD", *archive_paths)
    bundle = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(archive)) as incoming, zipfile.ZipFile(bundle, "w") as out:
        for item in incoming.infolist():
            out.writestr(item, incoming.read(item.filename))
        out.writestr("SOURCE_COMMIT", commit + "\n")
    payload = base64.b64encode(bundle.getvalue()).decode()
    if task == "smoke":
        command = ["-m", "src.train.kaggle_entry", "--fast"]
    else:
        if task in ("state-farm", "state-farm-pretrained"):
            data_root = "/kaggle/input"
        if task == "rt-bene":
            data_root = "/kaggle/temp/rt-bene/prepared"
        command = [
            "-m",
            "src.train.phase1",
            "--data-root",
            str(data_root),
            "--output",
            "/kaggle/working/artifacts/phase1",
            "--splits",
            "/kaggle/working/artifacts/phase1_splits.json",
            "--device",
            device,
        ]
        if task in ("state-farm", "state-farm-pretrained"):
            config_path = (
                "configs/state_farm_pretrained.yaml"
                if task == "state-farm-pretrained"
                else "configs/state_farm.yaml"
            )
            command.extend(
                [
                    "--config", config_path,
                    "--manifest",
                    "/kaggle/temp/state-farm/manifest.csv",
                ]
            )
            output_name = (
                "state_farm_pretrained"
                if task == "state-farm-pretrained"
                else "state_farm"
            )
            command[command.index("--output") + 1] = (
                f"/kaggle/working/artifacts/{output_name}"
            )
            command[command.index("--splits") + 1] = (
                "/kaggle/working/artifacts/state_farm_splits.json"
            )
        if fast:
            command.append("--fast")
        if task == "rt-bene":
            command.extend(["--manifest", data_root + "/manifest.csv"])
    data_setup = ""
    if task == "rt-bene":
        data_setup = (
            "subprocess.run([sys.executable, '-m', 'src.data.fetch_rt_bene', "
            "'--output', '/kaggle/temp/rt-bene/source'], cwd=code, check=True)\n"
            "subprocess.run([sys.executable, '-m', 'src.data.rt_bene', "
            "'--source', '/kaggle/temp/rt-bene/source', '--output', "
            "'/kaggle/temp/rt-bene/prepared'], cwd=code, check=True)\n"
        )
    if task in ("state-farm", "state-farm-pretrained"):
        data_setup = (
            "subprocess.run([sys.executable, '-m', 'src.data.state_farm', "
            "'--data-root', '/kaggle/input', '--output', "
            "'/kaggle/temp/state-farm/manifest.csv'], cwd=code, check=True)\n"
        )
    if task == "state-farm-pretrained":
        split_document = load_baseline_splits()
        split_payload = base64.b64encode(
            zlib.compress(json.dumps(split_document, separators=(",", ":")).encode("utf-8"))
        ).decode("ascii")
        data_setup += (
            "import zlib\n"
            "split_path = Path('/kaggle/temp/state-farm/baseline_splits.json')\n"
            "split_path.parent.mkdir(parents=True, exist_ok=True)\n"
            f"split_path.write_bytes(zlib.decompress(base64.b64decode({split_payload!r})))\n"
        )
        command[command.index("--splits") + 1] = "/kaggle/temp/state-farm/baseline_splits.json"
    bootstrap = (
        "import base64, io, zipfile, subprocess, sys\n"
        "from pathlib import Path\n"
        "code = Path('/kaggle/temp/project-code')\n"
        "code.mkdir(parents=True, exist_ok=True)\n"
        f"zipfile.ZipFile(io.BytesIO(base64.b64decode({payload!r}))).extractall(code)\n"
        f"{data_setup}"
        f"subprocess.run([sys.executable, *{command!r}], cwd=code, check=True)\n"
    )
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "cells": [
            {
                "cell_type": "code",
                "id": "integration-smoke",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": bootstrap.splitlines(keepends=True),
            }
        ],
    }
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "run.ipynb").write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    shutil.copyfile(ROOT / "kaggle/kernel-metadata.json", stage / "kernel-metadata.json")
    metadata = json.loads((stage / "kernel-metadata.json").read_text())
    metadata["enable_gpu"] = device == "cuda"
    metadata["enable_internet"] = task in ("rt-bene", "state-farm-pretrained")
    if task == "rt-bene":
        metadata["competition_sources"] = []
    if task in ("state-farm", "state-farm-pretrained"):
        metadata["competition_sources"] = ["state-farm-distracted-driver-detection"]
    (stage / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    (stage / "source.json").write_text(json.dumps({"commit": commit}), encoding="utf-8")
    print(f"Prepared committed source {commit} in {stage}")
    return commit


def cli(*args):
    executable = Path(sys.executable).parent / (
        "kaggle.exe" if sys.platform == "win32" else "kaggle"
    )
    if not executable.exists():
        raise RuntimeError("Install the local integration extra: pip install -e '.[dev,kaggle]'")
    result = subprocess.run([str(executable), *args], cwd=ROOT, capture_output=True, text=True)
    # Kaggle 2.2.5 can print missing-auth guidance and still return exit code 0.
    if "Authentication required to call the Kaggle API" in result.stdout + result.stderr:
        raise RuntimeError("Kaggle CLI needs local sign-in: .venv/Scripts/kaggle auth login")
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode:
        raise RuntimeError(f"Kaggle command failed (exit {result.returncode}); see message above.")
    if "Next page token:" in result.stdout:
        raise RuntimeError("Output has more pages; download is incomplete. Use Kaggle pagination.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "submit", "status", "fetch-results", "pull"])
    parser.add_argument("--version", type=int, help="Saved Kaggle version to fetch")
    parser.add_argument(
        "--task",
        choices=["smoke", "phase1", "rt-bene", "state-farm", "state-farm-pretrained"],
        default="smoke",
    )
    parser.add_argument("--data-root", help="Attached Kaggle path containing original MRL images")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    meta = json.loads((ROOT / "kaggle/kernel-metadata.json").read_text())
    kernel = meta["id"]
    if args.action == "prepare":
        prepare(task=args.task, data_root=args.data_root, fast=args.fast, device=args.device)
    elif args.action == "submit":
        prepare(task=args.task, data_root=args.data_root, fast=args.fast, device=args.device)
        extra = ["--accelerator", "NvidiaTeslaT4"] if args.device == "cuda" else []
        cli("kernels", "push", "-p", str(STAGE), *extra)
    elif args.action == "status":
        cli("kernels", "status", kernel)
    elif args.action == "pull":
        cli("kernels", "pull", kernel, "-p", str(ROOT / ".cache/kaggle-pull"), "-m")
    else:
        if args.version is not None:
            kernel += f"/{args.version}"
        # Unique destination preserves outputs from earlier runs.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        dest = ROOT / "results/kaggle" / stamp
        cli("kernels", "output", kernel, "-p", str(dest), "--page-size", "200")
        print(f"Downloaded saved outputs to {dest}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        sys.exit(str(exc))
