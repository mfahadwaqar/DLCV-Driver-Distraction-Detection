# One notebook for local data and optional training

Open `notebooks/driver_monitoring_training.ipynb` in VS Code. The default is a read-only preflight: it checks the kernel, prepared manifests, image paths, labels and saved subject splits. It does not train or change a Kaggle notebook. Existing notebooks and outputs remain available.

For the prepared upload ZIPs and exact connection checklist, follow
[Kaggle setup steps](KAGGLE_SETUP_STEPS.md). Local data preparation is already
complete on this computer; the preparation command below is for recovery/reuse.

## 1. Prepare the local data once

From the repository root in PowerShell, use the project Python environment:

```powershell
.venv/Scripts/python.exe scripts/prepare_local_data.py --datasets state_farm rt_bene
```

The preparation command uses already acquired local raw data. It does not download raw datasets. It writes canonical prepared images/manifests and fixed subject splits, validates source data and records reports. It preserves completed preparation and existing split assignments; a changed dataset or incompatible split must be investigated rather than silently reshuffled. Read the command's terminal report if preparation is blocked.

The notebook expects this layout beneath the repository (or a remote bundle root):

```text
data/
  state_farm/
    prepared/
      manifest.csv
      ... labeled training images ...
    preparation_report.json
  rt_bene/
    prepared/
      manifest.csv
      ... labeled left-eye crops ...
    preparation_report.json
  splits/
    state_farm_splits.json
    rt_bene_splits.json
```

CSV image paths are relative to their own prepared root. CSV subject IDs are loaded into the canonical dataset namespace by `src.data.manifest.load_manifest`. The split JSON contains these loaded rows, their fingerprint, seed, fractions and subject assignment. Transfer the manifests and splits together without editing or regenerating them. Preparation reports provide acquisition and integrity details; notebook path checks alone do not re-decode every image.

## 2. Run local preflight

In VS Code, select the repository `.venv` Python kernel. If it is not offered, install Jupyter/ipykernel support in that environment using VS Code's kernel prompt. For a new environment, install the project dependencies with `python -m pip install -e '.[phase1]'`; the project declares Python 3.11. Package installation is a separate manual setup step, never performed by the notebook.

Leave `RUN_TRAINING = False` and run all cells in order. Confirm the printed Python executable, source root and prepared-data paths. The notebook locates the repository from its working directory; if necessary, set `LOCAL_REPO_ROOT` explicitly. Check each task's `ready` status and the selected task's `READY` line. Missing PyYAML blocks manifest preflight; missing training packages are listed separately and block training.

The notebook reads existing split files and rejects missing images, unexpected datasets, split fingerprint/config differences, subject leakage, duplicate samples and missing train/validation class support. It never creates a new split. If a check fails, fix or finish explicit preparation, then rerun preflight.

## 3. Choose one supported task

Set the controls near the top of the notebook:

| Control | Purpose |
| --- | --- |
| `TASK = "state_farm"` | Five project distraction classes, using `configs/state_farm_pretrained.yaml` |
| `TASK = "rt_bene"` | Open versus closed/blinking eyes, using `configs/phase1.yaml` |
| `RUN_TRAINING = False` | Inspect only; the default |
| `RUN_TRAINING = True` | Explicitly start the selected task in step 5 |
| `DEVICE` | `cpu`, or `cuda` with a working GPU kernel |
| `FAST_RUN` | Small training smoke run; keep false for the configured full run |

State Farm labels map to safe driving, phone use, eating/drinking, other object and passenger interaction. The existing dataset mapping has limitations; these class names do not establish coverage of every real-world activity. RT-BENE eye closure is one signal, not a complete drowsiness diagnosis. Gaze off road, yawning and head nodding remain unsupported until appropriate labels are ready. DMD is reserved for held-out testing and is never a training option.

The pretrained State Farm model can fetch torchvision's official weights on its first training run; pre-cache them if the chosen kernel has no Internet. There is no raw-data downloader in this notebook.

## 4. Optional remote GPU with the same local notebook

The notebook file can stay open locally in VS Code while its selected Jupyter kernel executes elsewhere. The remote Python process cannot read your laptop's `E:` drive. A remote Jupyter connection does **not** upload repository code or prepared data.

Before using a Kaggle kernel:

1. Finish local preparation and preflight. Keep the exact manifest and split JSON files.
2. Create a readable source snapshot containing `src/`, `configs/`, `pyproject.toml`, this notebook and optionally a `SOURCE_COMMIT` text file identifying the revision. Include the intended local edits; do not export only an older Git revision. Exclude `.venv`, credentials, raw data and unrelated outputs. The snapshot is ordinary files, not an encoded bootstrap.
3. Manually transfer that snapshot and the selected prepared dataset(s), split files and preparation reports to storage available to the remote kernel. Preserve the layout above. Respect the datasets' access/redistribution terms; use private storage for restricted data. No Kaggle dataset or notebook is created or updated by this workflow.
4. In your own Kaggle notebook session, attach the source and prepared-data inputs, enable a GPU if desired, and ensure required Python packages are available. An attached source root might be `/kaggle/input/my-source/project`; a prepared bundle root might be `/kaggle/input/my-prepared-data` containing `data/`. Use the actual paths shown by that session.
5. If your setup provides a reachable authenticated Jupyter server for that session, connect VS Code through its Jupyter server connection UI and select that kernel. Kaggle's browser notebook URL itself is not a Jupyter server endpoint. The notebook does not start a server, tunnel, install an extension or discover credentials. If no supported Jupyter endpoint is available, upload/open this same notebook in Kaggle's browser editor instead.
6. Set `REMOTE_REPO_ROOT` and `REMOTE_BUNDLE_ROOT` in step 1. Leave training disabled and run preflight. Confirm it prints `Kaggle remote` and paths under the remote inputs. The code explicitly imports the selected source root; restart the kernel if another source snapshot was already imported.
7. Once ready, set `DEVICE = "cuda"` and deliberately enable training. There are no tokens or secrets in these cells. Never paste Jupyter tokens into the notebook or commit them.

An ordinary non-Kaggle remote machine is handled like a local filesystem: set `LOCAL_REPO_ROOT` to its source root and place its prepared data under that root. Kaggle detection is based on `/kaggle/working`.

## 5. Keep each experiment together

Training writes a new unique folder under `results/experiments/` locally or `/kaggle/working/experiments/` remotely. Existing results are protected. The root contains the effective config, `run_provenance.json`, `training.log` and final process status; the shared trainer writes checkpoint, metrics, predictions, environment details and used splits under `model/`.

`run_provenance.json` records the Git revision when present, visible source/config SHA-256 hashes, manifest/split hashes, split fingerprint and command. These hashes identify an uncommitted or copied snapshot, while a revision alone may not. No repository `SOURCE_COMMIT` file is rewritten. The trainer's older summary may say `local-unbundled`; use the notebook's provenance for the full source identity.

Step 6 reads results without training. Set `RESULTS_RUN` to an earlier experiment folder to inspect it. Use validation metrics for model choices and reserve the test split for final evaluation. Fast smoke runs and per-frame scores do not establish event-level drowsiness performance.

After a remote run, manually download the entire experiment folder before ending the session, then place it under local `results/experiments/` if desired. Keep source snapshot/provenance, checkpoint, config, splits, metrics, predictions and logs together. Kaggle working storage is temporary unless you explicitly save or download its outputs.
