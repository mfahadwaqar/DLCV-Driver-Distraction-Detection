# Local-first downloads and training handoff

Updated 2026-10-31. The selected downloads and local preparation are complete;
**do not download these datasets again**. State Farm has 22,424 prepared images
and RT-BENE has 107,350 prepared eye crops, all decoded successfully. All six
selected DMD archives passed integrity checks, and 126 unique NITYMED videos are
verified. NITYMED participant identities and event intervals still need annotation.

Continue with the [local notebook guide](LOCAL_NOTEBOOK.md) and
`notebooks/driver_monitoring_training.ipynb`. Keep `RUN_TRAINING = False` for
inspection. See [preparation evidence](LOCAL_PREPARATION_STATUS.md).

This workflow supersedes earlier instructions to download raw data inside Kaggle.
Keep original data, preprocessing code and notebooks on this computer. The download
steps below are recovery instructions for a missing source file or a new machine;
they are not outstanding tasks on this computer. Do not start a remote training
run as part of downloading.

NITYMED update: all 130 links in the current 720p listing have now been processed
automatically. The 126 unique MP4s are already in `data/nitymed/source/`; do not
download them again. All files passed full decoding, dimensions and local hash
checks. See [the measured download inventory](NITYMED_DOWNLOAD.md). Participant
identity and event annotation preparation still remain.

Project root:

```text
E:\Downloads\AIT\Third Semester\Deep Learning for Computer Vision\Project\Driver-Drowsiness-Detection
```

All destinations below are relative to that root. Keep original filenames and
compressed archives. No extraction or manual reorganisation is needed yet. The
`data/` directory is ignored by Git; never commit raw data or signed access links.

## 1. State Farm: download the complete official package

1. Open the [official competition data page](https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/data).
2. Sign in and accept the competition access/rules if Kaggle requests it.
3. Choose **Download All**, rather than an individual class or a third-party
   pre-split dataset. This is the complete competition download; the unlabelled
   competition test images are not supervised training examples.
4. Save the downloaded ZIP, normally named
   `state-farm-distracted-driver-detection.zip`, in:

   ```text
   data/state_farm/source/
   ```

Later preparation will inspect the outer/nested ZIP structure, use labelled
`train/c0` through `train/c9`, and preserve `driver_imgs_list.csv` for driver-wise
splits. Do not flatten images into a single directory or invent labels for the
competition test set. A small CSV already present elsewhere is not the full data.

## 2. RT-BENE: all usable original subject archives and labels

Official source: [RT-BENE, Zenodo record 3685316](https://zenodo.org/records/3685316).
The complete record lists about 937 MB. Download the subject table and the
original labels/archives for all 16 usable subjects; discarded `s006` is not needed.

From PowerShell at the project root, run the existing local downloader:

```powershell
.\.venv\Scripts\python.exe -m src.data.fetch_rt_bene --output data/rt_bene/source
```

This fetches and verifies the official size and MD5 for the subject table and all
**16 usable subjects**. Valid existing files are reused; a corrupt existing file
causes an error. Subject `s006` is deliberately excluded by the current command
because the official table marks it discarded.

Final destination:

```text
data/rt_bene/source/rt_bene_subjects.csv
data/rt_bene/source/s000_blink_labels.csv
data/rt_bene/source/s000_noglasses_eyes.tar
... through s016, excluding discarded s006
```

There should be **33 original files**. Keep the tar files compressed. Preparation
will exclude `s006` and disagreement labels (`0.5`), retain subject identities,
and use verified image/label matches. This dataset supplies eye-state supervision;
it does not supply yawning, nodding or gaze-off-road labels.

If the project virtual environment is unavailable, use the record's individual
Download links for the table and all 16 usable subject pairs in the same destination. Do not substitute a mirror.

## 3. DMD: three matched subjects, external test only

Use the access emails already granted to your academic account. Access links are
signed and short-lived; keep them out of Git, notebook cells and chat. Open the
README from each relevant email and retain a local copy with its dataset name.

Save all selected archives, unchanged, in:

```text
data/dmd/incoming/
```

| Package | Email selection | Exact archive filename | Action |
| --- | --- | --- | --- |
| Drowsiness | File 1 | `dmd-dataset-drowsiness-gA-1.tar.gz` | Already present and integrity-verified; do not download another copy |
| Drowsiness | File 2 | `dmd-dataset-drowsiness-gA-5.tar.gz` | Download |
| Drowsiness | File 3 | `dmd-dataset-drowsiness-gB-6.tar.gz` | Download |
| Distraction RGB | File 1 | `dmd-dataset-distraction-gA-1.tar.gz` | Download |
| Distraction RGB | Find by filename; numbering not verified | `dmd-dataset-distraction-gA-5.tar.gz` | Download if offered in the granted inventory |
| Distraction RGB | Find by filename; numbering not verified | `dmd-dataset-distraction-gB-6.tar.gz` | Download if offered in the granted inventory |

Inspect a link's target filename before downloading; do not assume File 2/3 mean
the same subjects in every email. If a matching distraction archive is absent,
record that gap rather than replacing it with a different subject silently. The
goal is subjects **1, 5 and 6** in both packages, subject to the actual release.

The existing subject-1 drowsiness archive is 828,325,787 bytes, SHA-256:

```text
a5b1d475452373e22fd0a99174fdec4ab509d694de6316c2f69b2b28908092b4
```

Its duplicate elsewhere adds no coverage. Leave existing copies alone. See
[the intake report](DMD_INTAKE.md) for what was actually inspected.

If access has expired, request renewed access through the official
[Drowsiness portal](https://opendatasets.vicomtech.org/di21-dmd-dataset-drowsiness/99e2fbd7)
or [Distraction RGB portal](https://opendatasets.vicomtech.org/di21-dmd-dataset-distraction-rgb-ir/a268f10c).
Use the granted email route while it works; no new account is needed merely to
repeat these steps. A separate [Gaze & Hands package](https://opendatasets.vicomtech.org/di21-dmd-dataset-gaze/8433939e)
can be inspected later if gaze evaluation is required; the six selected archives
do not establish gaze coverage.

**DMD is test only.** Keep every clip/view/session of these subjects out of
training, validation, threshold tuning and checkpoint selection. Archive names
alone do not prove annotation coverage. Later intake must inspect videos, JSON
intervals and supported classes before defining the external evaluation.

## 4. NITYMED: full-face candidate, annotation still required

Open the [official NITYMED site](https://datasets.esdalab.ece.uop.gr/) and reuse
the dataset access already granted. If collecting the full candidate source,
download **all available HDTV720 clips in all four sections**:

- Yawning → Male → HDTV720
- Yawning → Female → HDTV720
- Microsleep → Male → HDTV720
- Microsleep → Female → HDTV720

Save originals under `data/nitymed/source/`, using these four section subfolders
to preserve provenance and prevent filename collisions. Keep each original MP4
name. Retain a local inventory mapping section, official entry/link and filename.
The official site estimates the full 720p collection at approximately 700 MB;
use the actual granted inventory to check completeness. Full HD is an alternative
resolution, not an additional mandatory download. Existing sample videos can stay
where they are until duplicate checking during intake.

These clips remain **candidate source/development data** until actual participant
identities are linked across recordings and event intervals are annotated.
Clip numbers are not verified participant IDs. Do not label every frame of a
yawning or microsleep clip as that event: talking, normal and look-away intervals
also occur. Annotate supported positives/negatives and unknown intervals before
driver-wise splitting. Head-nod coverage and gaze labels remain unverified.

YawDD is optional, not part of the immediate required download list. If original
authorized videos are acquired later, retain them in `data/yawdd/source/` and
their participant metadata alongside them; access and event/identity preparation
still need to be resolved.

## What happens after downloads

**Your manual work now:** complete official browser downloads, run the RT-BENE
command locally (or download those files manually), and preserve original names
in the folders above. No Kaggle upload or remote run is required at this stage.

**Code work next:** inventory originals; check checksums and archive/video
readability; inspect labels; prepare derived data locally without modifying
originals; create versioned manifests and subject-wise splits; then create the
local training notebook and run a small local sanity check. Video identity and
interval annotation requires evidence and human review; it cannot be inferred
reliably from filenames. DMD remains separated throughout.

The next notebook will be a local `.ipynb` opened in VS Code. When it later uses
a remote Kaggle runtime, its Python process runs on Kaggle: it **cannot read
this computer's `E:` drive**. A prepared dataset bundle must be copied separately
to an authorized remote input location, and the notebook must select that remote
path. Keep code/notebook source local, record the bundle version, and transfer
only data needed by that run. Any source-specific transfer restrictions still
apply; private hosting alone does not establish permission.

Download completion establishes availability, not full project readiness. State
Farm and RT-BENE support initial distraction/eye-state baselines. Full temporal
yawning/nodding supervision, gaze-off-road training and complete external label
coverage remain explicit preparation/scope gaps until verified.

