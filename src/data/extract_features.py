"""Extract landmark features from a video without saving frames."""

import argparse
import csv
import math
from pathlib import Path

import cv2

from src.data.landmarks import FaceFeatures


def extract(video, model, output, subject, clip):
    if ":" not in subject:
        raise ValueError("Provide namespaced true subject ID, e.g. yawdd:participant01")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError("Video has no usable frame rate")
    features = FaceFeatures(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "subject_id",
                    "clip_id",
                    "timestamp",
                    "ear",
                    "mar",
                    "pitch",
                    "label_drowsiness",
                    "split",
                ],
            )
            writer.writeheader()
            index, last_ms = 0, -1
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp_ms = max(last_ms + 1, round(index * 1000 / fps))
                values = features.extract(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), timestamp_ms)
                writer.writerow(
                    {
                        "subject_id": subject,
                        "clip_id": clip,
                        "timestamp": timestamp_ms / 1000,
                        **values,
                        "label_drowsiness": -1,
                        "split": "",
                    }
                )
                index, last_ms = index + 1, timestamp_ms
    finally:
        features.close()
        cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--clip", required=True)
    args = parser.parse_args()
    extract(args.video, args.model, args.output, args.subject, args.clip)


if __name__ == "__main__":
    main()
