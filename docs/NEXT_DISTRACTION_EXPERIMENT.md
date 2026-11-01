# Next distraction experiment: regularized MobileNet

Prepared 2026-11-01. No new research training has run for this candidate.

The completed baseline achieved validation accuracy 66.04% and macro-F1 63.15%.
Training loss fell near zero while validation macro-F1 varied substantially.
This suggests overfitting to training drivers. The next candidate changes the
training recipe while keeping MobileNetV3-Small, the five mapped classes and the
exact 18/3/5 train/validation/test driver memberships. An 85% accuracy goal remains
unmet and is not promised by these changes.

## One candidate to run

`configs/state_farm_regularized.yaml` specifies:

- ImageNet initialization and 224-pixel inputs, batch size 32, seed 42.
- Mild position/scale/rotation and brightness/contrast/saturation augmentation on
  training images only. Validation preprocessing remains the baseline resize and
  normalization. The driver recipe does not mirror images.
- AdamW, learning rate 0.0001, weight decay 0.01, cosine learning-rate schedule.
- At most 20 epochs; stop after six epochs without improved validation macro-F1.
- Select `best.pt` by validation macro-F1. Existing training-only class weights
  remain unchanged.
- `evaluate_test: false`: test images are not loaded or scored in this candidate
  run. Historical split metadata remains intact. DMD is never included.

These are a combined recipe change, not an isolated ablation. Accept it only if
validation results improve sufficiently; inspect accuracy and each class as well
as macro-F1. Phone-use validation recall and passenger-interaction validation F1
were weak in the baseline. Do not select recipes using repeated test scores.
The baseline test set has already been reported, so it is not an untouched future
benchmark; keep DMD held out for later external evaluation after the method freezes.

The APIs used are documented in the official
[RandomAffine](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.RandomAffine.html),
[ColorJitter](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.ColorJitter.html),
and [AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
references. The chosen parameter values are our experimental hypothesis.

## Manual Kaggle setup for this run

1. The prepared-data dataset is already uploaded. Keep its version, manifests and
   splits unchanged.
2. Open the existing private **DLCV Project Code** dataset, choose its **New Version**
   upload action, and upload `.cache/kaggle_setup/dlcv-project-code-v2.zip`.
   Keep it private and retain `project/` as the internal root.
3. In the project Kaggle notebook, update/reattach that code input to its new
   version. Restart the GPU session after the old session has finished; reconnect
   VS Code using the current VS Code Compatible URL. Keep Internet enabled for
   pretrained weights. The existing prepared-data input remains attached.
4. Reopen the local `notebooks/driver_monitoring_training.ipynb`. Its default is
   `RUN_TRAINING = False`; run inspection first. Step 4 must show the regularized
   config, `evaluate_test: false`, and the new source revision. Missing that config
   means the runtime still has the previous code dataset version.
5. Once both datasets are READY and the GPU is available, deliberately set
   `RUN_TRAINING = True`, keep `TASK = "state_farm"` and `FAST_RUN = False`, and
   run all cells once. There is one candidate, not a hidden sweep.
6. Wait through the final **Step 7**. It should print **MODEL BACKUP ATTACHED**.
   Then press **Ctrl+S in VS Code**. This saves the experiment ZIP as a notebook
   output attachment on this computer. The Python source remains readable; the
   attachment contains binary model artifacts. Do not clear outputs before saving.
7. Also download the printed experiment-specific `_backup.zip` from Kaggle when
   practical. Verify the local backup before ending the runtime. A local saved
   notebook without Step 7's attachment is not a model backup.

To recover the saved attachment locally:

```powershell
.venv/Scripts/python.exe -m src.train.notebook_backup notebooks/driver_monitoring_training.ipynb
```

The restore command verifies SHA-256, ZIP integrity and path containment. It
refuses to replace an existing experiment folder. A restored experiment contains
the checkpoint, effective configs, metrics, predictions, splits, provenance and
training log. The backup is limited to a 50 MiB ZIP / 512 MiB expanded contents;
larger experiments require a separate downloaded archive.

## Verification completed locally

35 focused tests passed; two Windows symlink tests were skipped because creating
symlinks is unavailable on this host. Ruff passed. The synthetic training
integration test guards every dataset read and fails if any test image is opened
when `evaluate_test` is false; it also verifies the selected checkpoint loads in
the existing demo. Historical/default training behavior remains covered by tests.

All current notebook code cells executed with actual prepared data and training
disabled: both datasets READY in 5.3 seconds. A real 6,214,953-byte baseline
checkpoint was carried through a saved notebook output attachment and restored
byte-for-byte locally. This validates backup serialization and recovery; VS Code
must still save the attachment produced by the remote run. No new model accuracy
is claimed by these checks.

The previous completed notebook is preserved under ignored
`results/notebook-backups/completed-baseline-75bb72c0e95d.ipynb`.
