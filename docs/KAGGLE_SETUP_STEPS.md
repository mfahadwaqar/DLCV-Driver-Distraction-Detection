# Connect the local notebook to Kaggle

Use this checklist after the verified local preparation. All original data stays
on this computer. Upload copies of the two prepared ZIPs below; no raw archive or
video upload is needed for these two image-training tasks.

## 1. Upload the two prepared packages

The packages are in the project's `.cache/kaggle_setup/` folder:

| Local ZIP | Kaggle dataset title | Contents |
| --- | --- | --- |
| `dlcv-project-code.zip` | `DLCV Project Code` | Readable source, configs, notebook and source revision |
| `dlcv-prepared-training.zip` | `DLCV Prepared Training` | Both prepared image datasets, manifests, fixed splits and preparation reports |

Open [Kaggle's dataset uploader](https://www.kaggle.com/datasets/new).
Upload the code ZIP, enter its title, select **Private**, and create the dataset.
Repeat for the training-data ZIP, also **Private**. Wait until processing finishes.
Kaggle unpacks uploaded ZIPs; retain the internal directory structure.

Check that the code dataset contains `project/src/train/phase1.py`, and the data
dataset contains `data/state_farm/prepared/manifest.csv` and
`data/rt_bene/prepared/manifest.csv`. If the ZIP is still processing, wait before
connecting the runtime. The local `transfer_inventory.json` records package sizes,
SHA-256 hashes and the code revision. Do not upload that inventory separately.

## 2. Attach both inputs to the existing Kaggle notebook

Use the existing project notebook in the DLCV browser group:
[open the Kaggle editor](https://www.kaggle.com/code/mfahadwaqar2002/dlcv-driver-drowsiness-detection/edit).
Keep it private. Under **Add Input**, find your two new private datasets and add
both. A previously attached raw competition dataset alone is not the prepared
bundle used by the local notebook.

Enable **Internet** for the pretrained model's official initial weights and choose
**Settings > Accelerator > GPU T4 x2** (or another available GPU). The current
trainer uses one GPU; selecting two does not automatically distribute training.

Start the session. A minimal cell such as `print("Kaggle runtime ready")` can start
it without running earlier training code. Do not use Run All on an old training
notebook just to start the server.

## 3. Connect VS Code to that session

In Kaggle, choose **Run > Kaggle Jupyter Server**. With the session running, copy
the **VS Code Compatible URL**. Use it only in the VS Code connection prompt;
do not put the token-bearing URL in source files or Git.

In VS Code, open the local `notebooks/driver_monitoring_training.ipynb`. Select
the kernel at the top right, then **Select Another Kernel > Existing Jupyter
Server**. Enter the URL, give it a recognizable name, and choose its Python kernel.
The Kaggle browser notebook is the runtime anchor; the readable cells you execute
are in the local VS Code notebook. The files remain local while Python runs remotely.

## 4. Enter the remote input paths

In the first code cell, keep inspection mode and enter the input roots:

```python
RUN_TRAINING = False
TASK = "state_farm"
FAST_RUN = False
DEVICE = "cuda"
REMOTE_REPO_ROOT = "/kaggle/input/dlcv-project-code/project"
REMOTE_BUNDLE_ROOT = "/kaggle/input/dlcv-prepared-training"
```

These are example mount paths for the requested dataset names. Use the actual
paths in Kaggle's Input file tree if different. Some sessions use paths containing
`/kaggle/input/datasets/<owner>/<dataset>/`. The source root must directly contain
`src/` and `configs/`; the bundle root must directly contain `data/`. Do not enter
an `E:` path or a Kaggle webpage URL.

The notebook imports the uploaded source snapshot directly. Do not run
`pip install -e .` on Kaggle: the local project's Python version constraint may
differ from Kaggle's runtime. Its ordinary training dependencies are checked in
step 2 of the notebook; report any missing-package message before training.

## 5. Verify the connection without training

Run all cells in the local VS Code notebook with `RUN_TRAINING = False`.
Expected output:

- `Kernel: Kaggle remote` and remote paths under `/kaggle/input/`.
- No missing training packages.
- State Farm: `ready: true`, 22,424 images.
- RT-BENE: `ready: true`, 107,350 images.
- `Selected task: READY` and `Training skipped`.

The image-path check may take a few minutes. If anything is blocked, keep training
off and share the error text. There is no need to re-download the original data.
At this point, tell the coding agent **"Kaggle inspection passed"**. The setup is
complete. This package contains the existing baseline configurations; an upload
does not itself create a new accuracy-improvement experiment.

The private prepared-data dataset can be reused across sessions. Later source
changes require a new code snapshot/dataset version; editing a local `.py` file
does not update the uploaded copy automatically. Keep the recorded source revision
with each experiment.

## 6. When a training run is agreed

Choose one task, set `RUN_TRAINING = True`, and run the cells in order once.
Each task starts a separate experiment; avoid running it simultaneously in the
browser. Do not rerun training just to inspect its results. After completion, set
`RUN_TRAINING = False` and use the printed experiment path as `RESULTS_RUN` for
inspection. A disconnected/stopped session does not provide automatic resume.

Before ending the session, run this additional cell after all training has stopped:

```python
from pathlib import Path
import shutil

experiments = Path("/kaggle/working/experiments")
assert experiments.is_dir(), "No experiment outputs exist yet."
archive = shutil.make_archive(
    "/kaggle/working/driver_monitoring_results", "zip", root_dir=experiments
)
print("Download before stopping the session:", archive)
```

Refresh Kaggle's Output file list and download `driver_monitoring_results.zip`.
Save it locally under `results/experiments/` and retain it until the checkpoint,
metrics and logs are verified. Save the local notebook with Ctrl+S too.
Private dataset inputs persist separately from the running session; working
outputs need an explicit saved output or local download. VS Code's notebook
outputs are not a backup of the model files, and its edits are not automatically
mirrored to Kaggle's browser notebook. A browser Save & Run All action may execute
the browser's different cells rather than the local training workflow.

## References

- [Kaggle datasets: private uploads and automatic ZIP extraction](https://www.kaggle.com/docs/datasets).
- [Kaggle notebooks: Jupyter Server and VS Code connection](https://www.kaggle.com/docs/notebooks).
- [VS Code: select an existing Jupyter server](https://code.visualstudio.com/docs/datascience/jupyter-kernel-management#_existing-jupyter-server).
