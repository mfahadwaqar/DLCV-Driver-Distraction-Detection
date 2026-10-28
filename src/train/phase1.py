"""Train B2 on real, subject-disjoint samples. Run training on Kaggle."""

import argparse
import csv
import importlib.metadata
import json
import math
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.manifest import load_manifest
from src.data.mrl import ImageDataset, index_mrl
from src.data.schema import DISTRACTION, DROWSINESS
from src.data.splits import assert_no_leakage, fast_subset, saved_splits
from src.eval.metrics import classification_metrics, save_metrics
from src.models.baseline import build_baseline
from src.train.losses import masked_cross_entropy


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(2)


def image_transform(size, train=False):
    ops = [transforms.Resize((size, size))]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops.extend(
        [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    return transforms.Compose(ops)


def collate_samples(batch):
    return {
        "image": torch.stack([s["image"] for s in batch]),
        "label_drowsiness": torch.tensor([s["label_drowsiness"] for s in batch]),
        "label_distraction": torch.tensor([s["label_distraction"] for s in batch]),
    }


def predict(model, loader, device, class_ids, label_key="label_drowsiness"):
    truth, predictions, probabilities = [], [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            probs = model(batch["image"].to(device)).softmax(1).cpu()
            labels = encode_labels(batch[label_key], class_ids)
            valid = labels != -1
            truth.extend(labels[valid].tolist())
            predictions.extend(probs[valid].argmax(1).tolist())
            probabilities.extend(probs[valid].tolist())
    return truth, predictions, probabilities


def encode_labels(labels, class_ids):
    """Map supported project states to the single-task classifier's indices."""
    encoded = torch.full_like(labels, -1)
    for index, state in enumerate(class_ids):
        encoded[labels == state] = index
    if ((labels != -1) & (encoded == -1)).any():
        raise ValueError("Sample has a label outside configured class_ids")
    return encoded


def training_class_weights(rows, class_ids, device, label_key="label_drowsiness"):
    """Balance supervised classes using training subjects only."""
    counts = [sum(r[label_key] == state for r in rows) for state in class_ids]
    if any(count == 0 for count in counts):
        raise ValueError("Each configured class needs training samples")
    inverse_roots = [1 / math.sqrt(count) for count in counts]
    scale = len(counts) / sum(inverse_roots)
    weights = [value * scale for value in inverse_roots]
    return counts, weights, torch.tensor(weights, dtype=torch.float32, device=device)


def train(config, root, output, split_path, manifest=None, fast=False, device="cpu"):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory to preserve previous experiment results")
    label_key = config.get("label_key", "label_drowsiness")
    if label_key not in ("label_drowsiness", "label_distraction"):
        raise ValueError("Unknown classification target")
    taxonomy = DISTRACTION if label_key == "label_distraction" else DROWSINESS
    rows = load_manifest(manifest) if manifest else index_mrl(root)
    if any(r["dataset"].lower() == "dmd" for r in rows):
        raise ValueError("DMD cannot enter Phase 1 training")
    parts = saved_splits(rows, split_path, config["seed"], config["split_fractions"])
    if fast:
        parts = fast_subset(parts, config["fast_samples_per_split"], config["seed"], label_key)
    assert_no_leakage(parts)
    labels = config.get("class_ids", [0, 1])
    if (
        len(labels) < 2
        or len(set(labels)) != len(labels)
        or not set(labels) <= set(range(len(taxonomy)))
    ):
        raise ValueError("Configure at least two distinct project class_ids in taxonomy range")
    if not ({r[label_key] for r in rows} - {-1}) <= set(labels):
        raise ValueError("Dataset labels differ from configured class_ids")
    for split, group in parts.items():
        support = {r[label_key] for r in group} - {-1}
        if not support:
            raise ValueError(f"No supervised samples in {split}")
        if split in ("train", "val") and support != set(labels):
            raise ValueError(
                f"Missing class support in {split}; review subjects without test tuning"
            )
    seed_everything(config["seed"])
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select a Kaggle GPU")
    output.mkdir(parents=True, exist_ok=True)
    loaders = {}
    for name, group in parts.items():
        dataset = ImageDataset(root, group, image_transform(config["image_size"], name == "train"))
        loaders[name] = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=name == "train",
            num_workers=config["num_workers"],
            collate_fn=collate_samples,
            generator=torch.Generator().manual_seed(config["seed"]),
        )
    model = build_baseline(len(labels), config["pretrained"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    train_counts, weight_values, class_weights = training_class_weights(
        parts["train"], labels, device, label_key
    )
    print(
        json.dumps({"train_class_counts": train_counts, "class_weights": weight_values}), flush=True
    )
    best, history = -1.0, []
    names = [taxonomy[state] for state in labels]
    epochs = config["fast_epochs"] if fast else config["epochs"]
    if epochs < 1:
        raise ValueError("At least one epoch required")
    for epoch in range(epochs):
        model.train()
        loss_sum, count = 0.0, 0
        for batch in loaders["train"]:
            x = batch["image"].to(device)
            y = encode_labels(batch[label_key], labels).to(device)
            # MobileNet batchnorm needs >1 value per channel at the final feature map.
            if len(x) == 1 and config["image_size"] <= 32:
                raise ValueError("Increase image size or batch size for batch normalization")
            optimizer.zero_grad(set_to_none=True)
            loss = masked_cross_entropy(model(x), y, weight=class_weights)
            loss.backward()
            optimizer.step()
            valid = int((y != -1).sum())
            loss_sum += loss.item() * valid
            count += valid
        actual, predicted, _ = predict(model, loaders["val"], device, labels, label_key)
        val = classification_metrics(actual, predicted, names)
        history.append(
            {"epoch": epoch + 1, "train_loss": loss_sum / count, "val_macro_f1": val["macro_f1"]}
        )
        print(json.dumps(history[-1]), flush=True)
        if val["macro_f1"] > best:
            best = val["macro_f1"]
            torch.save(model.state_dict(), output / "best.pt")
    model.load_state_dict(torch.load(output / "best.pt", map_location=device, weights_only=True))
    reports = {}
    for split in ("val", "test"):
        actual, predicted, probs = predict(model, loaders[split], device, labels, label_key)
        report = classification_metrics(actual, predicted, names)
        report.update(
            {
                "split": split,
                "fast": fast,
                "dataset_scope": sorted({r["dataset"] for r in rows}),
                "task": "distraction"
                if label_key == "label_distraction"
                else "eye-state"
                if {r["dataset"] for r in rows} <= {"mrl", "rt_bene"}
                else "drowsiness",
            }
        )
        save_metrics(report, output, split)
        reports[split] = report
        with (output / f"{split}_predictions.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["label", "prediction", *[f"p_{i}" for i in labels]])
            writer.writerows(
                [labels[a], labels[p], *prob]
                for a, p, prob in zip(actual, predicted, probs, strict=True)
            )
    source = Path(__file__).resolve().parents[2] / "SOURCE_COMMIT"
    summary = {
        "training_performed": True,
        "fast": fast,
        "device": device,
        "class_ids": labels,
        "label_key": label_key,
        "unsupported_classes": [name for i, name in enumerate(taxonomy) if i not in labels],
        "train_class_counts": dict(zip(names, train_counts, strict=True)),
        "class_weights": dict(zip(names, weight_values, strict=True)),
        "source_commit": source.read_text().strip() if source.exists() else "local-unbundled",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "checkpoint_mb": (output / "best.pt").stat().st_size / 1e6,
        "splits": {
            n: {"samples": len(g), "subjects": len({r["subject_id"] for r in g})}
            for n, g in parts.items()
        },
        "macro_f1": {n: r["macro_f1"] for n, r in reports.items()},
        "limitations": [
            "Per-frame classification only; no event-level or trucking validation",
            "No in-subject comparison or unseen-driver drop computed",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (output / "used_splits.json").write_text(json.dumps(parts, indent=2), encoding="utf-8")
    (output / "environment.txt").write_text(
        "\n".join(
            sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
        ),
        encoding="utf-8",
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/phase1.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    print(
        json.dumps(
            train(
                config,
                args.data_root,
                args.output,
                args.splits,
                args.manifest,
                args.fast,
                args.device,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
