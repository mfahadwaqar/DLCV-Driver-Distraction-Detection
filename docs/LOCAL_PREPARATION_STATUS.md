# Verified local preparation

Verified on 2026-10-31. This stage prepared existing local data and executed the
inspection notebook. It did not start training, download data, or change Kaggle.
Original archives and earlier model results remain in place.

## Ready for the two existing training tasks

| Dataset and purpose | Prepared images | Train / validation / test images | Drivers per split |
| --- | ---: | --- | --- |
| State Farm: five mapped distraction classes | 22,424 | 14,881 / 2,712 / 4,831 | 18 / 3 / 5 |
| RT-BENE: open versus closed/blinking eyes | 107,350 | 81,046 / 9,299 / 17,005 | 11 / 2 / 3 |

Every prepared image decoded successfully; both reports show zero missing and
zero corrupt images. RT-BENE excludes 6,787 disagreement labels and the officially
discarded subject s006. All 33 selected RT-BENE source files passed official
size/MD5 checks against cached Zenodo record 3685316. The complete State Farm ZIP
passed CRC checks for 102,152 members; only its labelled training images and driver
CSV were prepared. No official State Farm checksum was available locally.

The split verification compared every sample identity, label and driver assignment
with the historical experiment splits. All matched after path remapping and removal
of repeated dataset prefixes. No driver was reassigned and no new random split was
generated. Canonical loaded subjects use one prefix, such as `state_farm:p012`.

Local evidence (ignored by Git):

- `data/{state_farm,rt_bene}/preparation_report.json`: counts and integrity evidence.
- `data/{state_farm,rt_bene}/prepared/manifest.csv`: indexed prepared samples.
- `data/splits/{state_farm,rt_bene}_splits.json`: fixed sample/driver assignments.
- `data/local_preparation_resume.log`: completed preparation log.
- `.cache/local-notebook-preflight.log`: actual execution of every default code cell.

## Notebook verification

Open `notebooks/driver_monitoring_training.ipynb`; see [the guide](LOCAL_NOTEBOOK.md).
All seven code cells were executed sequentially in the project Python environment
with the notebook's default `RUN_TRAINING = False`. Both datasets reported ready,
all 129,774 referenced image paths existed, fingerprints and class support passed,
and training was skipped. No experiment folder was created by this execution.
This proves inspection readiness, not new model accuracy or a completed GPU run.

Focused preparation, archive-audit and existing dataset-adapter tests passed:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_prepare_local_data.py tests/test_local_video_audit.py tests/test_rt_bene.py tests/test_state_farm.py -q
.venv/Scripts/python.exe -m ruff check scripts/prepare_local_data.py scripts/audit_local_video_data.py tests/test_prepare_local_data.py tests/test_local_video_audit.py
```

Result: 15 tests passed; Ruff passed. New tests cover split/label preservation,
unknown drivers, incompatible existing image protection, selective ZIP extraction
and resume, saved-split compatibility, checksum failures, archive path containment,
gzip trailer corruption, metadata-only extraction and blank annotation identities.

## Video data and remaining work

All six selected DMD archives passed complete gzip CRC/trailer and tar traversal
checks. Only annotations/metadata were extracted for inspection. DMD remains
external test only; no model scoring or training was performed on it. Per-view
video timing and label alignment still need verification before final scoring.

NITYMED has 126 unique verified videos from 130 successful source-link records
(four duplicate-content records). Full decoding passed in the earlier download
verification; the later audit rehashed files and matched that evidence. These are
126 videos, not 126 established participants. No more downloads are needed for
this selected local intake. See [video audit details](LOCAL_VIDEO_DATA_STATUS.md).

Before NITYMED can support training or participant-independent event evaluation,
establish reliable identities across clips and annotate reviewed event/negative
intervals. The template is `data/nitymed/annotation_template.csv`; its identity and
event fields are intentionally blank. Do not infer event boundaries from website
categories. Yawning, nodding and gaze labels remain incomplete; the two ready image
tasks do not establish coverage of those proposal tasks.

Next, transfer a readable code snapshot and the selected prepared data bundle to
the intended GPU kernel, then rerun inspection before enabling training. A VS Code
remote-kernel connection does not transfer this computer's files automatically.
Keep manifests, preparation reports and saved splits together. No remote transfer
or new training was performed during this local preparation stage.
