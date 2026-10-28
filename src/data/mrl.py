"""Strict original-filename MRL adapter. Confirm format against acquired archive."""

import re
from pathlib import Path

from PIL import Image

from src.data.schema import Sample

# subject_image_gender_glasses_eye_reflection_lighting_sensor
FILENAME = re.compile(
    r"(?P<subject>s[0-9]{4})_(?P<image>[0-9]+)_(?P<gender>[01])_"
    r"(?P<glasses>[01])_(?P<eye>[01])_(?P<reflection>[012])_"
    r"(?P<lighting>[01])_(?P<sensor>0[123])\.png",
    re.IGNORECASE,
)


def parse_filename(name):
    match = FILENAME.fullmatch(Path(name).name)
    if not match:
        raise ValueError(f"Not an original MRL filename: {name}")
    values = match.groupdict()
    return {
        "subject_id": "mrl:" + values["subject"].lower(),
        "dataset": "mrl",
        "label_distraction": -1,
        # MRL 0=closed, 1=open; project 0=Alert, 1=Eyes closed.
        "label_drowsiness": 1 - int(values["eye"]),
        "metadata": {k: int(v) for k, v in values.items() if k != "subject"},
    }


def index_mrl(root):
    root = Path(root)
    files = sorted(root.rglob("*.png"))
    if not files:
        raise FileNotFoundError(
            "No MRL images found. Supply the original archive with subject IDs."
        )
    rows = []
    for path in files:
        row = parse_filename(path.name)
        row["path"] = path.relative_to(root).as_posix()
        rows.append(row)
    return rows


class ImageDataset:
    def __init__(self, root, rows, transform=None):
        self.root = Path(root).resolve()
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index) -> Sample:
        row = self.rows[index]
        path = (self.root / row["path"]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Image path escapes data root")
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {
            "image": image,
            "label_distraction": row["label_distraction"],
            "label_drowsiness": row["label_drowsiness"],
            "subject_id": row["subject_id"],
            "dataset": row["dataset"],
            "metadata": dict(row["metadata"]),
        }
