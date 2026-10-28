"""Package committed code, submit Kaggle runs, and fetch saved outputs."""

import argparse
import base64
import io
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / ".cache" / "kaggle-submit"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def prepare(stage=STAGE, task="smoke", data_root=None, fast=False, device="cpu"):
    if task == "phase1" and not data_root:
        raise ValueError("Phase 1 requires --data-root pointing to an attached Kaggle dataset")
    if git("status", "--porcelain").strip():
        raise RuntimeError("Commit changes first: runs must use a clean committed snapshot.")
    commit = git("rev-parse", "HEAD").decode().strip()
    # Explicit allowlist: never bundle data, credentials, results, or .git.
    archive = git("archive", "--format=zip", "HEAD", "src", "configs")
    bundle = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(archive)) as incoming, zipfile.ZipFile(bundle, "w") as out:
        for item in incoming.infolist():
            out.writestr(item, incoming.read(item.filename))
        out.writestr("SOURCE_COMMIT", commit + "\n")
    payload = base64.b64encode(bundle.getvalue()).decode()
    if task == "smoke":
        command = ["-m", "src.train.kaggle_entry", "--fast"]
    else:
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
        if fast:
            command.append("--fast")
    bootstrap = (
        "import base64, io, zipfile, subprocess, sys\n"
        "from pathlib import Path\n"
        "code = Path('/kaggle/working/project-code')\n"
        "code.mkdir(parents=True, exist_ok=True)\n"
        f"zipfile.ZipFile(io.BytesIO(base64.b64decode({payload!r}))).extractall(code)\n"
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
    parser.add_argument("--task", choices=["smoke", "phase1"], default="smoke")
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
