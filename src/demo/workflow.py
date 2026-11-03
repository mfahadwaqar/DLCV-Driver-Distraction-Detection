"""Local, unscored video integration of the two saved classifiers."""

import argparse
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import cv2
import yaml

from src.demo.run import run
from src.eval.frozen_distraction import SELECTION, read_json, sha256, within


def inspect_demo(project, source=None):
    """Verify artifacts and video metadata, without model predictions or file writes."""
    project = Path(project).resolve()
    config = read_json(project / "configs/video_demo.json")
    selected = read_json(project / SELECTION)
    distraction = within(project, selected["model_dir"])
    eye = within(project, config["eye_model_dir"])
    face = within(project, config["face_model"])
    checkpoints = [
        (distraction / "best.pt", selected["checkpoint_sha256"]),
        (eye / "best.pt", config["eye_checkpoint_sha256"]),
        (face, config["face_model_sha256"]),
    ]
    for path, digest in checkpoints:
        if sha256(path) != digest:
            raise ValueError(f"Saved model SHA-256 mismatch: {path}")
    distraction_config = yaml.safe_load((distraction / "config.yaml").read_text())
    eye_config = yaml.safe_load((eye / "config.yaml").read_text())
    if (
        any(
            distraction_config[key] != selected[key] for key in ("model", "class_ids", "image_size")
        )
        or distraction_config.get("label_key") != "label_distraction"
        or eye_config["model"] != config["eye_model"]
        or eye_config["image_size"] != config["eye_image_size"]
        or eye_config.get("class_ids", [0, 1]) != [0, 1]
        or eye_config.get("label_key", "label_drowsiness") != "label_drowsiness"
    ):
        raise ValueError("Saved model task/configuration differs")
    video = Path(source) if source else Path(config["development_source"])
    video = (project / video).resolve() if not video.is_absolute() else video.resolve()
    if not any(
        video.is_relative_to((project / base).resolve())
        for base in ("data/nitymed/source", "data/demo")
    ):
        raise ValueError(
            "Use development footage in data/nitymed/source or data/demo; keep DMD held out"
        )
    if not video.is_file():
        raise FileNotFoundError(f"Development video missing: {video}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("FFmpeg is needed to save the notebook-compatible preview")
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError("Cannot open development video")
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if not all(math.isfinite(v) and v > 0 for v in (fps, frames, width, height)):
            raise ValueError("Development video needs valid FPS, frame count and dimensions")
    finally:
        capture.release()
    return {
        "ready": True,
        "demo_only": True,
        "scoring_performed": False,
        "training_performed": False,
        "source": str(video),
        "source_sha256": sha256(video),
        "source_frames": int(frames),
        "source_fps": fps,
        "source_dimensions": [int(width), int(height)],
        "distraction_model": selected["model"],
        "distraction_checkpoint": str(distraction),
        "distraction_checkpoint_sha256": selected["checkpoint_sha256"],
        "eye_checkpoint": str(eye),
        "eye_checkpoint_sha256": config["eye_checkpoint_sha256"],
        "face_model": str(face),
        "face_model_sha256": config["face_model_sha256"],
        "ffmpeg": ffmpeg,
        "settings": {key: config[key] for key in ("confidence", "persist_seconds", "max_gap")},
    }


def record_development_usage(project, plan):
    """Retain evidence that preview footage must not become an unseen test sample."""
    path = Path(project) / "data/demo/development_usage.json"
    records = read_json(path) if path.exists() else []
    if not isinstance(records, list):
        raise ValueError("Preserving unrecognized development usage record; inspect it first")
    if not any(row["sha256"] == plan["source_sha256"] for row in records):
        records.append(
            {
                "path": str(Path(plan["source"]).relative_to(Path(project).resolve())),
                "sha256": plan["source_sha256"],
                "role": "qualitative_development_demo",
                "recorded_utc": datetime.now(timezone.utc).isoformat(),
                "identity_status": "not established here",
                "restriction": (
                    "Exclude this clip and its participant from unseen-test claims "
                    "until identity review"
                ),
            }
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def run_demo(project, output, max_frames=250, source=None):
    project = Path(project).resolve()
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Use an empty demo folder to preserve previous runs")
    if not isinstance(max_frames, int) or max_frames < 1:
        raise ValueError("Choose a positive maximum frame count")
    plan = inspect_demo(project, source)
    record_development_usage(project, plan)
    output.mkdir(parents=True, exist_ok=True)
    (output / "input_provenance.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    result = run(
        SimpleNamespace(
            source=plan["source"],
            eye_checkpoint=Path(plan["eye_checkpoint"]),
            distraction_checkpoint=Path(plan["distraction_checkpoint"]),
            face_model=Path(plan["face_model"]),
            events=output / "events.jsonl",
            device="cpu",
            max_frames=max_frames,
            no_display=True,
            output_video=output / "annotated_raw.mp4",
            frame_log=output / "frame_predictions.jsonl",
            summary=output / "processing_summary.json",
            **plan["settings"],
        )
    )
    expected = min(max_frames, plan["source_frames"])
    if result["processed_frames"] != expected:
        raise ValueError(
            "Video stopped before the expected frame count; preserving incomplete output"
        )
    preview = output / "annotated_preview.mp4"
    subprocess.run(
        [
            plan["ffmpeg"],
            "-v",
            "error",
            "-n",
            "-i",
            str(output / "annotated_raw.mp4"),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(preview),
        ],
        check=True,
    )
    capture = cv2.VideoCapture(str(preview))
    decoded = 0
    try:
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            decoded += 1
    finally:
        capture.release()
    if decoded != result["processed_frames"]:
        raise ValueError("Annotated preview failed full frame-count verification")
    for path, digest in [
        (Path(plan["distraction_checkpoint"]) / "best.pt", plan["distraction_checkpoint_sha256"]),
        (Path(plan["eye_checkpoint"]) / "best.pt", plan["eye_checkpoint_sha256"]),
        (Path(plan["face_model"]), plan["face_model_sha256"]),
        (Path(plan["source"]), plan["source_sha256"]),
    ]:
        if sha256(path) != digest:
            raise ValueError("An input changed during demonstration")
    result.update(
        demo_only=True,
        decoded_preview_frames=decoded,
        preview=str(preview),
        source_sha256=plan["source_sha256"],
        completed_utc=datetime.now(timezone.utc).isoformat(),
    )
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("VIDEO DEMO COMPLETE:", preview, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-frames", type=int, default=250)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps(inspect_demo(args.project, args.source), indent=2))
    elif args.output is None:
        parser.error("--output is required for a video demonstration")
    else:
        run_demo(args.project, args.output, args.max_frames, args.source)


if __name__ == "__main__":
    main()
