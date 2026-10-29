"""Run separate eye-state/distraction baselines on video or webcam with JSONL events."""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from src.data.landmarks import FaceFeatures
from src.data.schema import DISTRACTION
from src.models.baseline import build_baseline
from src.train.phase1 import image_transform


class Persistence:
    """Emit alert transitions only after uninterrupted high-confidence observations."""

    def __init__(self, seconds=1.0, max_gap=0.75):
        if seconds < 0 or max_gap <= 0:
            raise ValueError("Invalid persistence duration or maximum gap")
        self.seconds, self.max_gap = seconds, max_gap
        self.candidate, self.started, self.active, self.last = None, None, None, None

    def update(self, timestamp, label):
        if not math.isfinite(timestamp) or (self.last is not None and timestamp <= self.last):
            raise ValueError("Timestamps must strictly increase")
        gap = self.last is not None and timestamp - self.last > self.max_gap
        self.last = timestamp
        events = []
        if self.active is not None and (label != self.active or gap):
            events.append({"event": "end", "label": self.active, "timestamp": timestamp})
            self.active = None
        if label != self.candidate or gap:
            self.candidate, self.started = label, timestamp
        if label is not None and self.active is None and timestamp - self.started >= self.seconds:
            self.active = label
            events.append({"event": "start", "label": label, "timestamp": timestamp})
        return events


class Classifier:
    def __init__(self, folder, target, device="cpu"):
        self.folder = Path(folder)
        config = yaml.safe_load((self.folder / "config.yaml").read_text())
        self.class_ids = config.get("class_ids", [0, 1])
        if config.get("label_key", "label_drowsiness") != target:
            raise ValueError(f"Wrong checkpoint task in {folder}")
        expected = [0, 1] if target == "label_drowsiness" else [0, 1, 2, 3, 5]
        if self.class_ids != expected or config["model"] != "mobilenet_v3_small":
            raise ValueError("Unsupported model or class mapping")
        self.device = device
        self.model = build_baseline(len(self.class_ids), pretrained=False).to(device).eval()
        self.model.load_state_dict(
            torch.load(self.folder / "best.pt", map_location=device, weights_only=True)
        )
        self.transform = image_transform(config["image_size"])

    def predict(self, rgb):
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = self.model(tensor).softmax(1)[0]
        index = int(probabilities.argmax())
        return self.class_ids[index], float(probabilities[index])


def eye_bounds(points, width, height):
    """Pad a detected anatomical left-eye region; clamp it to the frame."""
    points = np.asarray(points)
    if points.shape != (6, 2) or not np.isfinite(points).all():
        return None
    center = points.mean(axis=0)
    span = float(np.ptp(points[:, 0]))
    if span < 4:
        return None
    x1, x2 = max(0, int(center[0] - span * 0.75)), min(width, int(center[0] + span * 0.75))
    y1, y2 = max(0, int(center[1] - span * 0.4)), min(height, int(center[1] + span * 0.4))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


class EyeCropper(FaceFeatures):
    def crop(self, rgb, timestamp_ms):
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect_for_video(image, timestamp_ms)
        if not result.face_landmarks:
            return None
        height, width = rgb.shape[:2]
        points = result.face_landmarks[0]
        eye = [(points[i].x * width, points[i].y * height) for i in (362, 385, 387, 263, 373, 380)]
        bounds = eye_bounds(eye, width, height)
        if bounds is None:
            return None
        x1, y1, x2, y2 = bounds
        return rgb[y1:y2, x1:x2]


def run(args):
    import cv2

    if args.events.exists():
        raise ValueError("Choose a new event file to preserve previous runs")
    if not 0 <= args.confidence <= 1:
        raise ValueError("Confidence must be in [0, 1]")
    torch.set_num_threads(2)
    eye = Classifier(args.eye_checkpoint, "label_drowsiness", args.device)
    distraction = Classifier(args.distraction_checkpoint, "label_distraction", args.device)
    cropper = EyeCropper(args.face_model)
    webcam = args.source.isdecimal()
    capture = cv2.VideoCapture(int(args.source) if webcam else args.source)
    if not capture.isOpened():
        capture.release()
        cropper.close()
        raise ValueError("Could not open video/webcam source")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not webcam and (not math.isfinite(fps) or fps <= 0):
        capture.release()
        cropper.close()
        raise ValueError("Video needs valid FPS for temporal inference")
    gates = {
        name: Persistence(args.persist_seconds, args.max_gap) for name in ("eye", "distraction")
    }
    args.events.parent.mkdir(parents=True, exist_ok=True)
    names = {"eye": ("Open eyes", "Closed eyes"), "distraction": DISTRACTION}
    start, frame_index, last_ms = time.monotonic(), 0, -1
    try:
        with args.events.open("x", encoding="utf-8") as log:
            log.write(
                json.dumps(
                    {
                        "event": "run",
                        "source": args.source,
                        "eye_checkpoint": str(args.eye_checkpoint),
                        "distraction_checkpoint": str(args.distraction_checkpoint),
                        "confidence": args.confidence,
                        "persist_seconds": args.persist_seconds,
                        "max_gap": args.max_gap,
                    }
                )
                + "\n"
            )
            while args.max_frames is None or frame_index < args.max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp = time.monotonic() - start if webcam else frame_index / fps
                timestamp_ms = max(last_ms + 1, int(timestamp * 1000))
                last_ms = timestamp_ms
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                crop = cropper.crop(rgb, timestamp_ms)
                predictions = {
                    "eye": eye.predict(crop) if crop is not None else (None, 0.0),
                    "distraction": distraction.predict(rgb) if crop is not None else (None, 0.0),
                }
                lines = []
                for task, (label, confidence) in predictions.items():
                    alert = (
                        label if label not in (None, 0) and confidence >= args.confidence else None
                    )
                    for event in gates[task].update(timestamp, alert):
                        record = {**event, "task": task, "class_name": names[task][event["label"]]}
                        if event["event"] == "start":
                            record["confidence"] = confidence
                        log.write(json.dumps(record) + "\n")
                        log.flush()
                        print(json.dumps(record), flush=True)
                    text = (
                        "Unknown" if label is None else f"{names[task][label]} ({confidence:.2f})"
                    )
                    lines.append(f"{task}: {text} | alert: {gates[task].active is not None}")
                frame_index += 1
                if not args.no_display:
                    for i, line in enumerate(lines):
                        cv2.putText(
                            frame,
                            line,
                            (10, 30 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 0, 255),
                            2,
                        )
                    cv2.imshow("Driver monitoring baseline demo - Q to quit", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            log.write(
                json.dumps(
                    {
                        "event": "run_end",
                        "processed_frames": frame_index,
                        "active_alerts": {name: gate.active for name, gate in gates.items()},
                    }
                )
                + "\n"
            )
    finally:
        capture.release()
        cropper.close()
        if not args.no_display:
            cv2.destroyAllWindows()
    print(f"Processed {frame_index} frames. Events: {args.events}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Video path or webcam index, e.g. 0")
    parser.add_argument(
        "--eye-checkpoint", type=Path, required=True, help="Folder with config.yaml/best.pt"
    )
    parser.add_argument("--distraction-checkpoint", type=Path, required=True)
    parser.add_argument("--face-model", type=Path, default=Path("data/models/face_landmarker.task"))
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--confidence", type=float, default=0.7)
    parser.add_argument("--persist-seconds", type=float, default=1.0)
    parser.add_argument("--max-gap", type=float, default=0.75)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--no-display", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
