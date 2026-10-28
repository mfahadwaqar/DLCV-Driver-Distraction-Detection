# GitHub / Kaggle workflow

Code repository: https://github.com/mfahadwaqar/DLCV-Driver-Distraction-Detection
Kaggle notebook: https://www.kaggle.com/code/mfahadwaqar2002/dlcv-driver-drowsiness-detection

Local Python 3.11 is for development. Kaggle's actual runtime is recorded per run.
Install locally: `.venv/Scripts/python.exe -m pip install -e ".[dev,kaggle]"`.
Authenticate with the official CLI's `kaggle auth login` on your own computer.
Never paste tokens into chat, commit them, or embed them in notebooks.

From the repository root (PowerShell):
```powershell
.\.venv\Scripts\python.exe scripts/kaggle_workflow.py prepare
.\.venv\Scripts\python.exe scripts/kaggle_workflow.py submit
.\.venv\Scripts\python.exe scripts/kaggle_workflow.py status
.\.venv\Scripts\python.exe scripts/kaggle_workflow.py fetch-results
```

Commit changes first. Prepare creates an ignored notebook bundle containing only
committed src/ and configs/ plus the source Git SHA. No GitHub token or network
clone is needed in Kaggle. Push code to GitHub normally for backup/review.
Submit updates the named Kaggle notebook and starts a saved CPU run. It replaces
its code with the generated entry notebook. Metadata retains official State Farm
input and private visibility. This initial entry point is an integration smoke
check, not a training model. Training will replace/extend src/train/kaggle_entry.py
in Phase 1. Do not interpret its metadata counts as model metrics.

The smoke run saves artifacts/smoke.json, config.json and environment.txt beneath
/kaggle/working. Future training must save checkpoints, measured metrics and config
there too. Saved run outputs can be retrieved by the CLI; unsaved interactive files
are not a reliable archive. Use fetch-results --version N to request a particular
saved version. Each fetch has a new timestamp folder so earlier runs stay intact.
The embedded source commit in smoke.json identifies the exact code snapshot.
Pull downloads notebook/metadata to .cache/kaggle-pull for inspection and never
overwrites local source files. Model weights and raw outputs remain ignored.

GPU training configuration is deferred until an actual model smoke test passes.
No automatic polling or scheduled sync is installed. Fetch results after a saved
run completes. This workflow does not upload any datasets or face images to GitHub.
Official reference: https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md
