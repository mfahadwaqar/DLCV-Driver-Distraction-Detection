"""Persisted subject-wise splits with fingerprint and leakage rejection."""

import hashlib
import json
import random
from pathlib import Path

NAMES = ("train", "val", "test")


def fingerprint(rows):
    encoded = json.dumps(sorted(rows, key=lambda r: (r["dataset"], r["path"])), sort_keys=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def assert_no_leakage(parts):
    subjects, paths = {}, {}
    for split, rows in parts.items():
        if split not in NAMES:
            raise ValueError(f"Unknown split: {split}")
        for row in rows:
            subject = row["subject_id"]
            if not subject or ":" not in subject:
                raise ValueError("Expected namespaced subject identity")
            if subject in subjects and subjects[subject] != split:
                raise ValueError(f"Subject leakage: {subject}")
            subjects[subject] = split
            key = (row["dataset"], row["path"])
            if key in paths:
                raise ValueError(f"Duplicate sample: {key}")
            paths[key] = split


def make_splits(rows, seed=42, fractions=(0.70, 0.15, 0.15)):
    if len(fractions) != 3 or any(f <= 0 for f in fractions) or abs(sum(fractions) - 1) > 1e-8:
        raise ValueError("Three positive split fractions must sum to one")
    subjects = sorted({r["subject_id"] for r in rows})
    if len(subjects) < 3:
        raise ValueError("At least three subjects required for train/val/test")
    random.Random(seed).shuffle(subjects)
    ntrain = min(len(subjects) - 2, max(1, int(len(subjects) * fractions[0])))
    nval = min(len(subjects) - ntrain - 1, max(1, int(len(subjects) * fractions[1])))
    assignment = {
        s: name
        for name, group in zip(
            NAMES,
            (subjects[:ntrain], subjects[ntrain : ntrain + nval], subjects[ntrain + nval :]),
            strict=True,
        )
        for s in group
    }
    parts = {
        name: sorted(
            [r for r in rows if assignment[r["subject_id"]] == name],
            key=lambda r: (r["dataset"], r["path"]),
        )
        for name in NAMES
    }
    assert_no_leakage(parts)
    return parts


def saved_splits(rows, path, seed=42, fractions=(0.70, 0.15, 0.15)):
    path = Path(path)
    signature = fingerprint(rows)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if (
            saved["fingerprint"] != signature
            or saved["seed"] != seed
            or saved["fractions"] != list(fractions)
        ):
            raise ValueError("Saved split configuration/data differ; use a new split file")
        parts = saved["splits"]
        assert_no_leakage(parts)
        if fingerprint([r for part in parts.values() for r in part]) != signature:
            raise ValueError("Saved split contents have changed")
        return parts
    parts = make_splits(rows, seed, fractions)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"seed": seed, "fractions": fractions, "fingerprint": signature, "splits": parts},
            indent=2,
        ),
        encoding="utf-8",
    )
    return parts


def fast_subset(parts, max_samples=32, seed=42, label_key="label_drowsiness"):
    if max_samples < 2:
        raise ValueError("Fast subset needs at least 2 samples")
    result = {}
    for name, rows in parts.items():
        rng = random.Random(seed + NAMES.index(name))
        # Class-balanced selection for smoke runs; never reassign subjects.
        groups = {}
        for row in rows:
            groups.setdefault(row[label_key], []).append(row)
        for group in groups.values():
            rng.shuffle(group)
        selected = []
        while len(selected) < max_samples and any(groups.values()):
            for label in sorted(groups):
                if groups[label] and len(selected) < max_samples:
                    selected.append(groups[label].pop())
        result[name] = selected
    assert_no_leakage(result)
    return result
