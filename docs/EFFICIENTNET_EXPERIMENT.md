# EfficientNet-B0 backbone comparison

Prepared on 2026-11-01. Real EfficientNet training/results are pending. The
existing MobileNet checkpoints and completed notebooks are preserved.

## Validation error review completed

The regularized MobileNet has 814 errors among 2,712 validation images. The saved
prediction labels/counts reproduce the confusion matrix, and the source-class
metadata agrees with the current five-class mapping. Per-driver accuracy is
83.12%, 68.54% and 54.05% for p016, p064 and p075 respectively.

`scripts/analyze_distraction_validation.py` generates a ranked error CSV and
three local contact sheets. Only validation images are opened. Reproduce with:

```powershell
.venv/Scripts/python.exe -m scripts.analyze_distraction_validation --artifacts results/experiments/20261101T080231Z_state_farm_17193824/model --image-root data/state_farm/prepared --output results/experiments/20261101T080231Z_state_farm_17193824/validation_analysis
```

Visual review of the 12 highest-confidence safe-as-passenger examples and 12
passenger-as-other examples shows similar driver poses across action categories
and many closely related frames from the same people. Several c0 examples have
drivers looking toward the camera despite their original safe-driving category.
This suggests visual/task ambiguity; it does not prove annotation errors or
independently establish gaze labels. The samples are deliberately selected errors,
not a representative random sample. No labels were changed. In particular, the
heterogeneous c5/c7/c8 group and unavailable gaze supervision remain limitations.

## Controlled next candidate

The current reference remains pretrained MobileNetV3-Small: validation accuracy
66.04%, macro-F1 63.15%. The regularized candidate reached 69.99% accuracy but
61.84% macro-F1 and did not beat the preselected balanced-class criterion.

`configs/state_farm_efficientnet.yaml` replaces the pretrained backbone with
**EfficientNet-B0**. It retains the original pretrained MobileNet recipe:
224-pixel inputs, batch size 32, seed 42, Adam at 0.0001 for 10 epochs, baseline
horizontal-flip augmentation, original training-only class weights, and fixed
18/3/5 driver membership. Optimizer, regularization and early stopping are not
changed simultaneously with the backbone. The backbone's associated ImageNet
pretraining also differs, so describe this as a comparison of pretrained models.

Test scoring is disabled. Select the checkpoint and compare models by validation
macro-F1, reporting validation accuracy, per-class scores and per-driver results
as well. No 85% result is promised. DMD stays held out. The test set has already
been reported for older baselines and cannot be described as untouched.

The classifier factory, trainer and demo now support both backbones. Inference
constructs an uninitialized architecture and loads the saved task checkpoint,
so it does not download ImageNet weights again. Pretrained training uses official
torchvision weights; see the [EfficientNet-B0 documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html).

## What to do on Kaggle

1. Open the existing private **DLCV Project Code** dataset. Upload
   `.cache/kaggle_setup/dlcv-project-code-v3.zip` through **New Version**.
   Keep the prepared-data input unchanged.
2. Update the notebook's code input to that version. Start a GPU session with
   Internet enabled for the official pretrained weights, and connect VS Code to
   its current Jupyter server.
3. Reopen the local `notebooks/driver_monitoring_training.ipynb`. Leave
   `REMOTE_REPO_ROOT = ""`: step 2 locates the EfficientNet source beneath the
   named code input, with or without Kaggle's ZIP-name wrapper. If multiple
   matching snapshots are present, it asks for an explicit path rather than
   selecting one silently.
4. With `RUN_TRAINING = False`, Run All. Check the GPU name, both tasks READY,
   `state_farm_efficientnet.yaml`, and `model: efficientnet_b0`. If the config
   still shows MobileNet, the new code/notebook has not been loaded correctly.
5. Set `RUN_TRAINING = True`, keep `TASK = "state_farm"`, `FAST_RUN = False`,
   `DEVICE = "cuda"`, and Run All once.
6. After Step 7 prints **MODEL BACKUP ATTACHED**, press **Ctrl+S**. Ask the coding
   agent to restore/verify the local attachment before ending the runtime.

With our five-class heads, the locally constructed models have 4,013,953
parameters for EfficientNet-B0 versus 1,522,981 for MobileNetV3-Small. Training cost
and CPU latency must be measured before choosing the final monitoring backbone.

## Local verification

48 focused tests passed, with two Windows symlink tests skipped because this host
cannot create those links. Ruff passed. Actual-data notebook inspection completed
with both tasks READY and training disabled.

Architecture factory tests verify the chosen weights enum, replaced five-class
head and rejection of unknown names. Synthetic training tests cover both real
backbone implementations, validation-only image access and checkpoint loading in
the demo. Additional tests cover validation-error alignment and source detection
for flat/wrapped mounts, old snapshots and ambiguity. No ImageNet weights were
downloaded during local tests. The full prepared-data notebook inspection passed
with training disabled. These checks do not establish new model accuracy.

The completed regularized notebook, including its model attachment, is preserved
as `results/notebook-backups/completed-regularized-5e3425354806.ipynb`.
