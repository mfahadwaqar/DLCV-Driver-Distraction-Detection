# Run the two saved models on video

Use `notebooks/driver_monitoring_video_demo.ipynb` in the local project environment.
This stage needs no training, Kaggle runtime, new download or data upload. The
checkpoints, face-landmark asset, FFmpeg and development sample are already local.

## Steps in VS Code

1. Open `notebooks/driver_monitoring_video_demo.ipynb`.
2. Select **Driver monitoring (local .venv)**, or **Select Another Kernel → Python
   Environments → this project's `.venv/Scripts/python.exe`**. Use the local kernel,
   as with the testing notebook.
3. Leave `RUN_DEMO = False`, `MAX_FRAMES = 250` and `SOURCE_VIDEO = ""`.
   Click **Run All**. Step 2 should show READY and both model tasks.
4. Change only `RUN_DEMO = True` and click **Run All** once. This processes the first
   250 frames (10 source seconds) of the default development clip. Progress prints
   after the first frame and every 50 frames. Let processing and video encoding
   finish, even if CPU processing is slower than playback time.
5. After **VIDEO DEMO COMPLETE**, watch the annotated video in Step 4, press
   **Ctrl+S**, and tell the coding agent the run is complete. All outputs are local.

The default sample is `data/nitymed/source/P1042748_720.mp4`: 1280×720, 25 FPS.
For another development recording later, put it under `data/demo/` and set
`SOURCE_VIDEO` to that project-relative path. Existing completed output is reused
only when its inputs, settings and frame limit agree. Otherwise the notebook stops
and preserves it; ask the agent to configure a separate output folder. Incomplete
outputs are also preserved. Do not delete or overwrite earlier runs.

## What runs

| Component | Input | Output |
| --- | --- | --- |
| Selected ResNet-18 | Whole RGB frame, saved 224-pixel preprocessing | Five State Farm distraction categories and prediction confidence |
| Existing MobileNetV3-Small | Detected left-eye crop, saved 96-pixel preprocessing | Open/closed-eye prediction and confidence |
| Face landmark detector | Video frame | Eye crop; unavailable crops remain Unknown |
| Persistence rule | Class/confidence over consecutive observations | Start/end transitions for sustained demonstration alerts |

The full-frame distraction classifier can still run when no eye crop is available.
The two classifiers remain separate; this is not the proposed shared multi-task
architecture. The eye model has not learned yawning, head nodding or gaze.
No model weights or source labels change during this run.

The default 0.7 confidence, one-second persistence and 0.75-second gap settings are
the existing demonstration defaults. They are not validated false-alert operating
points. One isolated closed-eye frame does not immediately start a sustained
alert. Model confidence is not a calibrated probability of driver danger.

The eye crop uses the existing [MediaPipe Face Landmarker adapter](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python)
and supplied local model asset. Crop-domain differences from RT-BENE and camera
differences from State Farm remain limitations; neither classifier's image-test
accuracy can be transferred directly to this clip.

## Outputs and interpretation

Files are saved under `results/video_demo/local_preview/`:

- `annotated_preview.mp4`: H.264 preview playable in the notebook.
- `annotated_raw.mp4`: original annotated encoder output, preserved locally.
- `frame_predictions.jsonl`: frame number, nominal video timestamp, eye availability,
  predictions/confidences, active alert states and processing latency.
- `events.jsonl`: run information and sustained-alert start/end transitions.
- `processing_summary.json`: processing counts, timing, coverage and limitations.
- `input_provenance.json`: source/checkpoint/asset hashes and demonstration settings.
- `summary.json`: completed status, preview verification and the collected summary.

The overlay contains predictions, not ground-truth labels. Eye-crop coverage is
not eye accuracy. Predicted alert counts are not false-alarm rates. There are no
accuracy, recall, event-F1 or time-to-ground-truth-event results from this clip.
No DMD footage is opened for this demonstration; the workflow rejects it.

Decode/inference latency includes frame decoding, landmarks and both classifiers,
but excludes model loading, overlays and output writes. Overall processing time
includes the processing loop/output work before final H.264 conversion. Saving at
the source FPS does not imply processing kept up with that FPS. Timestamps use
nominal frame index/FPS, not verified per-camera PTS; this is not the DMD alignment
or external evaluation pipeline.

The source is logged in `data/demo/development_usage.json` before inference.
`P1042748_720.mp4` has now been inspected as development footage. Keep it out of
unseen-test claims, and review participant identities before deciding whether
other NITYMED clips containing the same participant can enter any evaluation set.
No identity is inferred or fabricated by this workflow. All media/logs stay in
Git-ignored local folders.

## Verification before handoff

25 focused tests passed, including existing classifier/persistence tests and
frozen-checkpoint evaluation tests. Ruff passed. The notebook check now accepts
a legitimately enabled, saved testing notebook; the user's executed notebooks
were preserved.

An actual-model smoke run processed 25 source frames, produced predictions from
both checkpoints, found eye crops in all 25 and fully decoded the 25-frame H.264
output. Source, checkpoint and landmark-asset hashes remained unchanged. This
proves integration, not accuracy. In that short run, median decode/inference was
82.33 ms/frame and p95 was 90.95 ms/frame; the source period is 40 ms, so this check
did not demonstrate real-time 25-FPS CPU processing. No threshold tuning followed.
Smoke artifacts are in `.cache/video-demo-smoke/`.

Every default notebook cell passed in the local Jupyter kernel with `RUN_DEMO`
false. The user-facing 250-frame output folder has not been created by setup.

Next, inspect the longer preview/logs, then prepare labeled external evaluation.
DMD needs stream/timestamp/label alignment and a frozen evaluation protocol.
NITYMED event training still needs reviewed identities and event intervals; the
demo does not complete that annotation work or the full proposal.
