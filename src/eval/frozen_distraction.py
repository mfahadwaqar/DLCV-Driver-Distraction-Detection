"""Evaluate the frozen State Farm checkpoint without training or model selection."""

import argparse
import csv
import hashlib
import json
import platform
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.manifest import load_manifest
from src.data.mrl import ImageDataset
from src.data.schema import DISTRACTION
from src.data.splits import assert_no_leakage, fingerprint
from src.demo.run import Classifier
from src.eval.metrics import classification_metrics, save_metrics
from src.train.phase1 import collate_samples

SELECTION = "configs/selected_distraction_checkpoint.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def within(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative or ":" in relative:
        raise ValueError("Expected a relative path inside the project")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Path escapes project root")
    return resolved


def inspect_inputs(project, selection_path=None):
    """Check frozen artifacts and test paths; do not decode images or run inference."""
    project = Path(project).resolve()
    selection = read_json(selection_path or project / SELECTION)
    if selection["selection_metric"] != "validation_macro_f1":
        raise ValueError("Checkpoint must have been selected on validation only")
    model_dir = within(project, selection["model_dir"])
    manifest = within(project, selection["manifest"])
    split_path = within(project, selection["splits"])
    for path, expected in [
        (model_dir / "best.pt", selection["checkpoint_sha256"]),
        (manifest, selection["manifest_sha256"]),
        (split_path, selection["split_sha256"]),
    ]:
        if sha256(path) != expected:
            raise ValueError(f"Frozen SHA-256 mismatch: {path.name}")
    config = yaml.safe_load((model_dir / "config.yaml").read_text(encoding="utf-8"))
    for key in ("model", "image_size", "class_ids"):
        if config[key] != selection[key]:
            raise ValueError(f"Frozen model configuration differs: {key}")
    if config.get("label_key") != "label_distraction" or config.get("evaluate_test") is not False:
        raise ValueError("Expected a validation-selected distraction training run")
    summary = read_json(model_dir / "summary.json")
    history = read_json(model_dir / "history.json")
    best = max(history, key=lambda row: row["val_macro_f1"])
    if (
        not summary["training_performed"]
        or summary["fast"]
        or summary["test_evaluated"]
        or summary["model"] != selection["model"]
        or summary["best_epoch"] != selection["best_epoch"]
        or best["epoch"] != selection["best_epoch"]
        or best["val_macro_f1"] != selection["validation_macro_f1"]
        or summary["macro_f1"]["val"] != selection["validation_macro_f1"]
    ):
        raise ValueError("Training history does not match the frozen validation selection")
    fixed = read_json(split_path)
    parts = fixed["splits"]
    assert_no_leakage(parts)
    rows = load_manifest(manifest)
    if (
        {row["dataset"] for row in rows} != {"state_farm"}
        or parts != read_json(model_dir / "used_splits.json")
        or fixed["fingerprint"] != selection["split_fingerprint"]
        or fingerprint(rows) != fixed["fingerprint"]
        or fingerprint([row for group in parts.values() for row in group]) != fixed["fingerprint"]
    ):
        raise ValueError("Expected the exact original State Farm manifest and fixed splits")
    test = parts["test"]
    if (
        len(test) != selection["test_samples"]
        or len({row["subject_id"] for row in test}) != selection["test_subjects"]
        or {row["label_distraction"] for row in test} != set(selection["class_ids"])
    ):
        raise ValueError("Frozen test sample/subject counts or class support differ")
    image_root = manifest.parent.resolve()
    for row in test:
        if not within(image_root, row["path"]).is_file():
            raise FileNotFoundError(f"Missing test image: {row['path']}")
    return {
        "ready": True,
        "selection": selection,
        "model_dir": str(model_dir),
        "manifest": str(manifest),
        "split_path": str(split_path),
        "image_root": str(image_root),
        "test_samples": len(test),
        "test_subjects": selection["test_subjects"],
        "training_performed": False,
        "inference_performed": False,
    }


def evaluate(project, output, device="cpu", batch_size=32, selection_path=None):
    """Score test once per new output folder; no optimizer or training call exists."""
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Use an empty output directory to preserve earlier test results")
    if device not in {"cpu", "cuda"} or batch_size < 1:
        raise ValueError("Choose cpu/cuda and a positive batch size")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use cpu for local testing")
    plan = inspect_inputs(project, selection_path)
    selected = plan["selection"]
    torch.set_num_threads(2)
    classifier = Classifier(plan["model_dir"], "label_distraction", device)
    rows = read_json(plan["split_path"])["splits"]["test"]
    dataset = ImageDataset(plan["image_root"], rows, classifier.transform)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_samples
    )
    names = [DISTRACTION[i] for i in selected["class_ids"]]
    lookup = {state: index for index, state in enumerate(selected["class_ids"])}
    truth, predicted, probabilities = [], [], []
    started = time.perf_counter()
    print(f"Testing frozen {selected['model']}: {len(rows):,} frames on {device}", flush=True)
    with torch.inference_mode():
        for step, batch in enumerate(loader, 1):
            probs = classifier.model(batch["image"].to(device)).softmax(1).cpu()
            truth.extend(lookup[int(label)] for label in batch["label_distraction"])
            predicted.extend(probs.argmax(1).tolist())
            probabilities.extend(probs.tolist())
            if step == 1 or step % 20 == 0 or step == len(loader):
                print(
                    f"Test: {len(truth):,}/{len(rows):,} frames "
                    f"({time.perf_counter() - started:.1f}s)",
                    flush=True,
                )
    if len(truth) != len(rows):
        raise ValueError("Incomplete test inference; refusing a partial report")
    if sha256(Path(plan["model_dir"]) / "best.pt") != selected["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed during evaluation")
    report = classification_metrics(truth, predicted, names)
    report.update(
        split="test",
        task="distraction",
        dataset_scope=["state_farm"],
        fast=False,
        accuracy=sum(a == b for a, b in zip(truth, predicted, strict=True)) / len(truth),
        training_performed=False,
    )
    subjects = defaultdict(lambda: {"images": 0, "correct": 0})
    for row, actual, prediction in zip(rows, truth, predicted, strict=True):
        counts = subjects[row["subject_id"]]
        counts["images"] += 1
        counts["correct"] += int(actual == prediction)
    per_driver = {
        subject: {**counts, "accuracy": counts["correct"] / counts["images"]}
        for subject, counts in sorted(subjects.items())
    }
    summary = {
        "training_performed": False,
        "test_evaluated": True,
        "model": selected["model"],
        "experiment": selected["experiment"],
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "accuracy": report["accuracy"],
        "macro_f1": report["macro_f1"],
        "test_samples": len(rows),
        "test_subjects": len(per_driver),
        "device": device,
        "inference_seconds": time.perf_counter() - started,
        "per_driver": per_driver,
        "limitations": [
            "State Farm per-frame distraction only; no gaze or drowsiness scoring",
            "Fixed test benchmark previously scored for MobileNet baselines",
            "No checkpoint, threshold or recipe selection from these test scores",
            "Batch inference duration is not live-video latency",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    save_metrics(report, output, "test")
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["path", "subject_id", "label", "prediction"]
            + [f"p_{i}" for i in selected["class_ids"]]
        )
        for row, actual, prediction, probs in zip(
            rows, truth, predicted, probabilities, strict=True
        ):
            writer.writerow(
                [
                    row["path"],
                    row["subject_id"],
                    selected["class_ids"][actual],
                    selected["class_ids"][prediction],
                    *probs,
                ]
            )
    provenance = {
        **plan,
        "inference_performed": True,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "batch_size": batch_size,
        "evaluator_sha256": sha256(Path(__file__)),
    }
    for name, value in [
        ("summary.json", summary),
        ("frozen_checkpoint.json", selected),
        ("evaluation_provenance.json", provenance),
    ]:
        (output / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps(inspect_inputs(args.project), indent=2))
    elif args.output is None:
        parser.error("--output is required for evaluation")
    else:
        evaluate(args.project, args.output, args.device, args.batch_size)


if __name__ == "__main__":
    main()
