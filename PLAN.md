# Project plan

## Phase 0 and stop point
Create a local Python 3.11 environment, Git repository, folder skeleton, YAML config,
developer commands and version record. Run empty pytest suite and lint, then commit.
STOP for your review of this plan before Phase 1.

## Assumptions and limitations
- CPU tests/demo; GPU training optional. Seed 42. Saved subject-wise train/val/test
  manifests, target proportions 70/15/15 where subject counts permit. Never split frames.
- Namespace subject IDs by dataset; investigate cross-dataset identity overlap.
- MRL crops support open/closed-eye classification only. They do not establish
  yawning, nodding, full-face landmarks, PERCLOS, or event-level performance.
- B1 and temporal metrics require full-face video with suitable annotations.
- Missing labels are -1 and masked per head. Both outputs can fire together.
- DMD is held out: never train, tune thresholds, or select models on it.
- Requested State Farm mapping has no Gaze off road training class. Obtain another
  permitted training source or explicitly report that class unsupported.
- Public car-cab data provides no trucking validation.
- No data, face imagery, credentials, checkpoints or runtime logs enter Git.
- Phase 0 installs pytest, ruff, PyYAML only. ML stack is deferred until CPU/GPU
  build selection and package/API compatibility checks in Phase 1.
- --fast means deterministic subsampling within saved splits, never bypassing isolation.
- In-subject comparisons are separately labeled diagnostics and cannot contaminate
  primary subject-disjoint evaluation.

## Phases and gates
0. Setup: environment, skeleton, empty-suite verification, lint, version log, commit.
1. Drowsiness: verify access/licenses and MRL filenames; schema/mock adapters; saved
   subject splits/leakage test; synthetic EAR/MAR tests; B1 landmarks/PERCLOS with
   validation-only threshold tuning; B2 per-frame CNN; held-out-subject evaluation.
2. Distraction: State Farm train-only mapping tests, subject splits, CNN baseline,
   per-class report. Preserve the exact requested c0-c9 mapping.
3. Shared backbone: two heads, masked weighted loss, EMA/majority smoothing,
   robustness augmentation, one-step CPU smoke test and single-task comparison.
4. Experiments: same-split baselines/ablations, glasses and low-light subsets,
   explicitly synthetic perturbations, held-out DMD shared-class evaluation.
5. Demo: FastAPI/WebSocket, webcam/video, persistent-state alerts, timestamp/state/
   confidence JSONL log, no stored frames by default, Docker, ONNX/CPU benchmark.
6. Report: measured JSON/CSV metrics, figures, limitations, reproduction commands.
Each phase ends with tests/lint, docs/phaseN.md and a Git commit. Never invent results.

## Inputs needed from you
1. Review this plan and authorize Phase 1 after setup.
2. Existing dataset locations or availability to complete access requests. Kaggle
   needs your account/competition acceptance. Keep credentials out of chat and Git.
3. Training hardware (NVIDIA model/VRAM, Colab or CPU) and disk/download budget.
   Ask before downloads over 5 GB or any global installation.
4. Grading rubric, final deadline, required report format.
5. Permitted full-face temporal data and a separate Gaze off road training source.
   Use clearly marked mock loaders while access is pending.

## Evaluation policy
Macro/per-class F1 and confusion matrices per head; event recall at fixed false
alerts/hour with explicit negative-duration and matching protocol; time to alert;
measured CPU latency/hardware/protocol; model size; optionally ECE. Crop accuracy
is not driver-state accuracy. Record seeds, manifests, configs and package versions.
Dataset access and licenses remain unverified at Phase 0.
