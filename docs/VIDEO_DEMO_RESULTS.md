# Completed two-model video demonstration

Completed on 2026-11-03 at 16:03:50 UTC. The user ran
`notebooks/driver_monitoring_video_demo.ipynb` locally on the CPU. The saved
notebook, video, logs, timing calculations and input hashes were verified after
completion without repeating model inference or changing weights/settings.

## Observed output

The source was the first 250 frames (10 source seconds) of NITYMED development
clip `P1042748_720.mp4`, 1280×720 at 25 FPS. Both saved models processed it:
the selected State Farm ResNet-18 and the RT-BENE MobileNetV3-Small eye classifier.

| Measurement | Observed value |
| --- | ---: |
| Processed / decoded annotated frames | 250 / 250 |
| Frames with an available eye crop | 250 / 250 |
| Eye predictions | Open eyes in all 250 frames |
| Distraction predictions | Talking to passenger in all 250 frames |
| Predicted closed-eye alert starts | 0 |
| Predicted distraction alert starts | 1, at source time 1.00 s |
| Processing-loop duration | 14.47 s |
| Effective processing throughput | 17.28 frames/s |
| Median decode + inference | 48.08 ms/frame |
| p95 decode + inference | 59.19 ms/frame |

The distraction alert stayed active through the preview cutoff. Its start at
one second agrees with the configured persistence rule; it is not a measured
time-to-ground-truth-event. No closed-eye alert does not prove the absence of eye
closure. Eye-crop availability is coverage, not eye-state accuracy.

The same passenger label across all frames is a generalization concern requiring
labeled review. The first/middle/last annotated frames were visually inspected;
they show changing facial/mouth poses, but visual snapshots do not independently
establish the five-class action ground truth. No frame was relabeled and no
threshold or classifier was tuned from this demonstration. A high model
confidence is not proof that the action label is correct.

## Interpretation and performance limits

The combined pipeline produces a playable annotated video, both streams of
predictions, persistence-based alert events and processing measurements. This
is an integration result, not an accuracy or external-test result. NITYMED has a
different camera/view distribution from State Farm; source image-test scores do
not establish accuracy on this clip. There are no reviewed event labels here.

At 17.28 FPS the measured loop did not keep up with the 25-FPS source. The saved
video still plays at source FPS, so smooth playback does not imply real-time
processing. Decode/inference timings include landmarks and both classifiers,
exclude model loading/overlays/output writes, and loop duration excludes final
H.264 transcoding. This is one CPU preview, not a controlled hardware benchmark.

## Verified and preserved artifacts

Output folder: `results/video_demo/local_preview/`.
Playable recording: `annotated_preview.mp4` (1,623,199 bytes).
Executed notebook backup:
`results/notebook-backups/completed-video-demo-0a3c3b3054a7.ipynb`.

Verification passed:

- Source, both checkpoints, landmark asset and settings match the inspected inputs.
- All 250 prediction rows have sequential frame IDs and nominal frame/FPS times;
  labels are in supported class sets and confidence values are finite/in range.
- Replaying the saved predictions through the same persistence rules reproduces
  the event log and every recorded active-alert state.
- Saved frame timings reproduce the reported median and p95 exactly.
- The H.264 recording fully decodes to 250 frames at the source FPS.
- The complete executed notebook is preserved separately. Original inputs and
  checkpoints are unchanged. No training or accuracy scoring was performed.

The development-usage record remains in `data/demo/development_usage.json`.
Keep this clip out of unseen-test claims and review participant identities before
assigning other clips of the same person to future NITYMED evaluation groups.
DMD was not opened or scored by this workflow.

## Next stage

Prepare labeled DMD external **frame-level** evaluation for the existing models:

1. Extract only the required local streams and supplied annotations; verify
   video decode counts, timestamps/FPS, camera shifts and terminal-frame handling.
2. Freeze camera choices, action-to-project mappings, ambiguity/visibility masks
   and sample accounting before model scoring. Keep DMD excluded from training,
   model selection, threshold selection and recipe tuning.
3. Evaluate distraction and supported open/closed-eye labels on the aligned
   samples; report each task and participant separately. No retraining is needed
   to begin this evaluation stage.

The demonstration thresholds are unvalidated operating points. Event-level
false-alert/time-to-alert claims need reviewed source validation video labels and
a frozen event-matching protocol first. NITYMED identities/event intervals remain
the manual annotation gap for yawning/microsleep training. Nodding and gaze
training supervision is still unresolved; these behaviors and the proposed shared
multi-task model have not been completed by the video demo.
