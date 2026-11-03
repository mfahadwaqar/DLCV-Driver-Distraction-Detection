# Additional models and dataset roles

Updated 2026-11-03. EfficientNet-B0 remains the best completed distraction
candidate: **86.50% validation accuracy / 81.18% validation macro-F1**. Its
checkpoint is restored locally. [EfficientNet-B2 completed](EFFICIENTNET_B2_RESULTS.md)
at 84.85% validation accuracy / 78.06% macro-F1, below B0. Its checkpoint is also
verified locally. ResNet-18 is prepared but has no project result yet.
A 90% result is a goal, not a prediction.

## These datasets do different jobs

| Dataset | Purpose | What has actually trained |
| --- | --- | --- |
| State Farm | Five mapped distraction classes | MobileNet variants, EfficientNet-B0 and B2; each trained on 14,881 images from 18 drivers |
| RT-BENE | Open versus closed/blinking eyes | A separate MobileNet model trained on 81,046 eye crops from 11 subjects |
| NITYMED | Potential yawning/microsleep video supervision | Videos verified locally; participant identities and reviewed event intervals are incomplete, so no trained event classifier yet |
| DMD | External held-out testing | Archives/annotations inspected; never used for training or tuning; model scoring still pending |

For State Farm, 2,712 images/3 drivers remain validation and 4,831 images/5 drivers
remain test. For RT-BENE, validation and test subjects also stay separate. Training
on every acquired image would remove those independent checks. The current
EfficientNet model does not combine these datasets or establish yawning, nodding,
gaze or complete drowsiness performance. Additional labels are needed for those
proposal tasks; downloading more copies of the existing datasets does not supply
them. The two ready image tasks can proceed independently of that annotation work.

## Two additional candidates

| Model | Input | Batch | Parameters with five-class head |
| --- | --- | --- | ---: |
| EfficientNet-B0 reference | 224 pixels | 32 | 4,013,953 |
| EfficientNet-B2 candidate | 288 pixels | 16 | 7,708,039 |
| ResNet-18 candidate | 224 pixels | 32 | 11,179,077 |

Both candidates use official ImageNet pretrained weights, Adam at 0.0001 for ten
epochs, the existing horizontal-flip training augmentation and training-only class
weights. Labels, seed, manifest and driver assignments stay unchanged. Validation
macro-F1 selects the checkpoint. Test scoring is disabled; DMD is excluded.

EfficientNet-B2's pretrained model uses a 288-pixel crop dimension in its
[official reference](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b2.html).
Our full-frame square-resize policy is retained, so this does not claim identical
official preprocessing. Batch size 16 reduces GPU memory demand. B2 changes
architecture, resolution and batch size together: this is a candidate comparison,
not an isolated architecture ablation. ResNet-18 retains B0's input/batch recipe;
see its [official implementation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
The pretrained weights differ across architectures as well.

Run this finite two-candidate comparison before making another recipe decision.
Compare accuracy, macro-F1 and safe-driving/passenger recall. Keep B0 if neither
candidate improves the chosen validation criterion. Do not use test scores to
pick a model. CPU latency and full event-system evaluation remain outstanding.

## Manual setup: one code upload, two selectable runs

Current position: B2 is complete and its saved notebook/model are backed up
locally. Continue with ResNet-18 in step 6; the v4 code and prepared-data inputs
do not need another upload. Steps 1–5 remain the original setup record.

1. In the existing private **DLCV Project Code** dataset, upload
   `.cache/kaggle_setup/dlcv-project-code-v4.zip` through **New Version**.
   The prepared-data input stays unchanged.
2. Update/attach that code input version, start a GPU session with Internet for
   official weights, and connect the local notebook to its current Jupyter server.
3. Reopen `notebooks/driver_monitoring_training.ipynb`. Keep
   `REMOTE_REPO_ROOT = ""` and inspect first:

   ```python
   RUN_TRAINING = False
   TASK = "state_farm"
   DISTRACTION_MODEL = "efficientnet_b2"
   FAST_RUN = False
   DEVICE = "cuda"
   ```

4. Run All and check the GPU, READY outputs and `model: efficientnet_b2` in Step 4.
   Then set `RUN_TRAINING = True` and Run All once to train B2.
5. Wait for Step 7's **MODEL BACKUP ATTACHED**, press Ctrl+S, and have the coding
   agent restore/verify the first experiment locally before changing the controls.
6. With the same uploaded code, change `DISTRACTION_MODEL = "resnet18"`, inspect
   with training false, then enable training and run once. Save and verify its
   Step 7 backup too. Every experiment has a distinct directory including the
   model name. Do not overwrite or discard the first saved model attachment.

Larger models can take longer and do not automatically improve generalization.
The notebook attachment can also be larger. Do not end the remote session until
the corresponding local model backup is verified.

The existing B0 and MobileNet choices remain available for inspecting their
configurations; completed runs need not be retrained. RT-BENE selection remains
its existing eye-state baseline and ignores the distraction-model selector.

## Verification

56 focused tests passed; two Windows symlink tests were skipped because this
host cannot create those links. Ruff passed. Tests cover all four real backbone
implementations, inference without pretrained downloads, architecture-specific
heads, validation-only access, checkpoint compatibility and notebook recovery.
Both new selections also passed full actual-data notebook inspection with
training disabled. These are implementation checks, not new accuracy results.

The completed B0 notebook including its attachment is preserved under ignored
`results/notebook-backups/completed-efficientnet-b0-e85a4f1203bc.ipynb`.
