# Test the selected model locally

Use `notebooks/driver_monitoring_testing.ipynb`. Keep the training notebook as
experiment history. This stage needs no Kaggle session, new dataset download,
model upload or training. The existing local CPU and saved ResNet-18 checkpoint
are sufficient.

## Steps in VS Code

1. Open `notebooks/driver_monitoring_testing.ipynb`.
2. Click **Select Kernel** at the top right, then **Select Another Kernel →
   Python Environments**, and choose this project's `.venv/Scripts/python.exe`.
   The installed kernel may also appear as **Driver monitoring (local .venv)**.
   Do not select the Kaggle Jupyter server for this notebook.
3. Leave the first cell at:

   ```python
   RUN_TESTING = False
   DEVICE = "cpu"
   BATCH_SIZE = 32
   ```

4. Click **Run All**. Step 2 should print `READY: True`, `resnet18`, epoch 10,
   4,831 test images and five test drivers. This inspection performs no inference.
   Stop and share the error if a hash, file or split check fails.
5. Change only `RUN_TESTING = True`, then click **Run All** once. Progress appears
   after the first batch and every 20 batches. CPU inference can take several
   minutes. Leave it running until `TEST COMPLETE` and the result tables appear.
6. Press **Ctrl+S**, then tell the coding agent it is complete. No result download
   is needed: the output files are already on this computer.

An already completed result is reused on a repeated Run All. An incomplete
nonempty result folder is preserved and causes an error; the agent should inspect
it rather than delete it or silently start another test. No other models are
selected or trained by this notebook.

## What gets tested and saved

The selected checkpoint is frozen in
`configs/selected_distraction_checkpoint.json`, including its exact SHA-256,
architecture, 224-pixel preprocessing size, five class IDs, validation selection,
manifest hash and fixed split hash. Inspection verifies these against local
artifacts and the saved training split. Test inference opens only test images,
uses evaluation mode and does not construct an optimizer or update weights.

The output folder is `results/testing/state_farm_resnet18_frozen/`:

- `test.json`: aggregate scores, per-class precision/recall/F1 and confusion matrix.
- `test.csv` and `test_confusion.csv`: tables for the report.
- `test_predictions.csv`: each test frame's path, driver, true/predicted label and
  class probabilities, in fixed split order.
- `summary.json`: accuracy, macro-F1, per-driver scores and task limitations.
- `frozen_checkpoint.json` and `evaluation_provenance.json`: model/input hashes
  and evaluation details.

Step 5 also creates `results/testing/state_farm_resnet18_frozen_reports.zip`.
It contains small reports and predictions; raw images and model weights are not
included. The model directory and original data are preserved. All generated
results remain ignored by Git.

This measures State Farm **per-frame distraction only**, not RT-BENE eye-state,
yawning, nodding, gaze, video events or DMD generalization. State Farm's fixed
test split was previously scored for the MobileNet baselines; it is not a
never-inspected dataset. Keep the selected model/recipe fixed after these scores.
Validation accuracy of 87.68% does not guarantee the same test accuracy.

## Environment and terminal alternative

The project environment needs the existing `phase1` packages plus the `notebooks`
extra. The local kernel has been installed for this workspace. For a new checkout:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[phase1,notebooks]"
```

The same checks/evaluation can run from the project terminal without a notebook:

```powershell
.\.venv\Scripts\python.exe -m src.eval.frozen_distraction --check-only
.\.venv\Scripts\python.exe -m src.eval.frozen_distraction --output results/testing/state_farm_resnet18_frozen
```

The CLI refuses a nonempty output folder. Use either the notebook or the terminal
for the actual test, not both. Reuse and inspect a completed report.

## Verification of this setup

Eleven focused tests passed; Ruff passed. Tests exercise real ResNet checkpoint inference on synthetic images,
reject accesses to non-test images and entry into training mode, reject changed
hashes/configuration/splits/validation selection, preserve checkpoint bytes, and
refuse result overwrite. Synthetic scores are not project measurements.
The actual-data inspection passed for 4,831 files and five drivers without scoring
them. Every default notebook cell also passed in the installed local Jupyter
kernel with testing disabled. The real test has not been run by setup.
