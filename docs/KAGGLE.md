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
Submit updates the named Kaggle notebook and starts a saved CPU run by default. It replaces
its code with the generated entry notebook. Metadata retains official State Farm
input and private visibility. The default task remains an integration smoke
check, not a training model. Phase 1 adds `--task phase1 --data-root ACTUAL_PATH`
and optional `--fast --device cuda` for MobileNet B2 training on acquired MRL data.
See phase1.md. Do not interpret integration metadata counts as model metrics.

The smoke run saves artifacts/smoke.json, config.json and environment.txt beneath
/kaggle/working. Future training must save checkpoints, measured metrics and config
there too. Saved run outputs can be retrieved by the CLI; unsaved interactive files
are not a reliable archive. Use fetch-results --version N to request a particular
saved version. Each fetch has a new timestamp folder so earlier runs stay intact.
The embedded source commit in smoke.json identifies the exact code snapshot.
Pull downloads notebook/metadata to .cache/kaggle-pull for inspection and never
overwrites local source files. Model weights and raw outputs remain ignored.

For `--device cuda`, metadata enables GPU and submission requests Kaggle T4.
Run --fast on real data first to verify the hosted package/runtime combination.
No automatic polling or scheduled sync is installed. Fetch results after a saved
run completes. This workflow does not upload any datasets or face images to GitHub.
Official reference: https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md

## Repeatable RT-BENE training

Use the local VS Code notebook for interactive checks. Its cells run on Kaggle, but
the local notebook is a separate file from the Kaggle browser editor. For a durable
training run, commit and push the code, then submit a saved Kaggle version with:

```powershell
.\.venv\Scripts\python.exe scripts/kaggle_workflow.py submit --task rt-bene --device cuda
.\.venv\Scripts\python.exe scripts/kaggle_workflow.py status
.\.venv\Scripts\python.exe scripts/kaggle_workflow.py fetch-results --version N
```

The saved run fetches RT-BENE from its official source into `/kaggle/temp`, prepares
the eye images there, and trains from subject-separated splits. The large source and
prepared images are temporary and will be fetched again on another run. Only the
checkpoint, predictions, metrics, splits, configuration, environment and summary go
under `/kaggle/working/artifacts` and become downloadable saved-version outputs.
Use the version number shown by Kaggle after `submit` for `fetch-results`. The local
results folder is timestamped and ignored by Git. Each run records its source commit.

The training loss uses class weights computed only from the training split to address
the heavy open-eye majority. Inspect per-class recall, confusion matrix and macro F1
on validation and test; the earlier tiny fast run was only an integration check.
The generated saved notebook is separate from the readable interactive notebook, so
submitting it does not replace local VS Code cells. The local notebook's uncommitted
outputs do not block submitting committed source code.
