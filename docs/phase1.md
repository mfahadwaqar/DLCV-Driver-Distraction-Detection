# Phase 1 — first eye-state result, broader evaluation pending

Started 2026-10-28 with user authorization. A real RT-BENE eye-state baseline now
has held-out subject results; see [the measured run](RT_BENE_RESULTS.md). Phase 1
remains incomplete for distraction, yawning, microsleep and full driver-state
evaluation. Eye-state accuracy is not a measured drowsiness-alert result.

## Implemented locally
- Strict candidate MRL filename parser, common image schema and explicit manifest
  adapter for requested-access full-face data. Original MRL schema still needs
  verification against an acquired archive. No native YawDD/NTHU loader is claimed.
- Seeded subject-disjoint splits, fingerprints, saved split-file reuse, duplicate
  rejection and leakage tests. --fast subsets never reassign subjects.
- B1: MediaPipe Face Landmarker, pixel-coordinate EAR/MAR, head-pitch transform,
  time-weighted sliding-window PERCLOS, sustained-state rules, gap/unknown handling,
  validation-only threshold search and held-out frame reporting. Rule thresholds
  are initial candidates, not calibrated findings. Simultaneous drowsiness signals
  use eyes-closed then yawning then nodding priority; compare on shared supported labels.
- B2: MobileNetV3-Small, single-task per-frame classifier, resize/normalization and
  training-only horizontal flip. No temporal stage or robustness augmentation.
  Random initialization by default; pretrained weights require explicit config.
- Missing-label masking, deterministic seeds, validation-based best checkpoint,
  one final held-out test evaluation, JSON/CSV reports, confusion matrices,
  predictions, model size, config, used split manifests and environment versions.
- Kaggle prepare/submit supports --task phase1, --data-root, --device and --fast.
  It still defaults to the previous integration smoke check.

## Checks actually performed
The local suite checks leakage/tampering, filename labels, image schema, path
boundary, rule geometry, blink/gap/missing handling, duration-weighted PERCLOS,
validation-only tuning, masked-loss gradients, one CPU optimizer step, and artifact
creation on artificial test pixels. Those artificial scores are not research results.
See results/phase1/status.json for the latest measured check count/time.
Official MediaPipe model initialized successfully; a blank frame returned None for
EAR/MAR/pitch, as expected. No human-face test footage was used.
Local CPU dependency versions are logged in environment-phase1-local.txt.

## Data and evaluation blockers
- Original MRL image/archive and current source terms are unavailable: official
  page and archive requests timed out. No original filename schema was inspected.
- YawDD participant-information ZIP received and nine-page PDF inspected. Public
  sharing permission varies at recording level. It does not establish a blanket
  permission to host video copies on Kaggle. Full video/labels not yet acquired.
- NTHU official endpoint failed; current access procedure remains unverified.
- Do not train from unavailable/assumed data, label whole yawning clips as yawning,
  or claim B1 can run on isolated MRL eye crops.
- Macro F1, B1/B2 comparison, glasses/low-light performance, event recall/false-alert
  rate/time-to-alert and unseen-driver drop are pending real-data evaluation.
  Event metrics need real annotations/negative durations; no approximation is made yet.

## Verified alternative data

The official RT-BENE `s012` sample was checksum-verified and prepared locally:
935 labeled left-eye images (912 open, 23 closed/blink), with 336 disagreement
labels excluded. It confirms the loader but contains only one subject. A private
Kaggle run can fetch the remaining subject archives directly from Zenodo and
check each file's official MD5/size. RT-BENE supports only the eye-state B2
baseline; it cannot validate full drowsiness or in-car alerts.

Four NITYMED videos were decoded and matched to their official yawning/microsleep
entries. Their clip numbers do not identify subjects or event intervals, so they
remain qualitative pipeline samples until annotations and identity mapping exist.

For the RT-BENE Kaggle eye-state smoke run, from a clean committed checkout:
```powershell
.\.venv\Scripts\python.exe scripts/kaggle_workflow.py submit --task rt-bene --device cuda --fast
```
The generated private notebook enables internet only for this official download.
After the run finishes, use `status` and `fetch-results` in the workflow script;
review held-out subjects and class support before reporting scores.

## Commands once data are available
Run locally to prepare acquired MRL data (replace placeholder paths):
```powershell
.\.venv\Scripts\python.exe -m src.data.prepare --data-root data/mrl/mrlEyes_2018_01 --splits data/splits/mrl.json --output results/phase1/data
```
Attach a permitted MRL source in kaggle/kernel-metadata.json dataset_sources first;
commit changes, then submit the first small run on Kaggle:
```powershell
.\.venv\Scripts\python.exe scripts/kaggle_workflow.py submit --task phase1 --data-root /kaggle/input/ACTUAL-SOURCE/mrlEyes_2018_01 --device cuda --fast
```
The input path is a placeholder, not an available dataset. After verifying the
small run, use a full run (omit --fast) with a new empty output directory. The saved
full-data split file is created before subsampling and included in Kaggle artifacts.
Compare fast/full split assignments and retain source versions/checksums. Attach
versioned split manifests for later experiments rather than silently creating new splits.

Full-face landmark extraction requires the official model asset and a real subject ID:
```powershell
.\.venv\Scripts\python.exe -m src.data.extract_features --video data/VIDEO --model data/models/face_landmarker.task --output data/features/CLIP.csv --subject yawdd:TRUE-ID --clip CLIP-ID
```
This stores features only and leaves labels=-1 and split blank. Assign verified
frame/interval labels and saved subject splits before scoring B1:
```powershell
.\.venv\Scripts\python.exe -m src.eval.rules --features data/features/annotated.csv --output results/phase1/b1
```
Do not combine unrelated clips into a single time series. Feature timestamps use
frame index and reported FPS, assuming constant-rate video; verify timing before
any event-level claims. The unknown-detection column counts failed detections as
errors in the four-state F1 and reports coverage separately.
