# Local baseline video/webcam demo

This prototype combines two separate MobileNetV3-Small checkpoints: RT-BENE
open/closed-eye classification and State Farm distraction classification. It is
not the proposed shared multi-task model. MediaPipe detects a face and crops the
anatomical left eye; the distraction model receives the full frame. The eye crop
method has not been validated on in-car video and may differ from RT-BENE crops.

Use downloaded result folders containing `best.pt` and `config.yaml`. The demo
loads each checkpoint's image size, ImageNet normalization and project class IDs.
Eye labels are open and closed; distraction supports safe driving, phone use,
drinking, other object and passenger. Gaze, yawning and nodding are unsupported.
One face is assumed; this prototype does not verify that it belongs to the driver.

From the project root with the phase1 dependencies installed:

```powershell
.\.venv\Scripts\python.exe -m src.demo.run --source "data/demo.mp4" --eye-checkpoint "results/kaggle/20261028T075933392791Z/artifacts/phase1" --distraction-checkpoint "PATH_TO_STATE_FARM_ARTIFACT_FOLDER" --events "results/demo/run01.jsonl"
```

Use `--source 0` for a webcam. Press Q to stop the display. Use `--no-display` for
headless processing of a video and `--max-frames 100` for a short software check.
Supply a new event filename for each run; existing logs are preserved. The official
`data/models/face_landmarker.task` asset is required. No video frames are saved.

The default alert requires one continuous second with class confidence at least
0.7. A changed class, missing eye/face observation or observation gap over 0.75
seconds resets persistence. `--persist-seconds`, `--confidence` and `--max-gap`
make these demonstration thresholds explicit. They have not been tuned or
validated against event annotations, so no false-alert rate or time-to-alert
performance is claimed. No face/usable eye crop means both predictions are Unknown.

JSONL records the run settings, start/end transitions and final active alerts.
Video timestamps use frame index divided by reported FPS; variable-rate files
need timing verification before any event evaluation. Webcam timestamps use
elapsed wall time. The display and log are qualitative demonstration outputs.
Never use the held-out presentation sample to adjust thresholds or train models.

Software verification: persistence resets on missing observations and long gaps;
eye crop bounds are clamped; wrong checkpoint tasks are rejected. The actual
RT-BENE checkpoint loaded and produced a valid binary prediction locally. A
three-frame blank video completed with the official MediaPipe asset, actual
RT-BENE weights and artificial distraction weights; both alerts stayed unknown.
This checks loading, video handling and event logging only. Real combined video
validation on representative driver footage has not been performed.

After the full State Farm artifact arrived, both actual baseline checkpoints
were loaded and checked on the same blank video. The no-face path produced no
alerts and completed the event log. Separate artificial-input forward passes
check each classifier's output mapping. These are software checks, not evidence
of driver-video accuracy; representative video validation remains pending.
