# EfficientNet-B0 validation results

Completed on 2026-11-01. Experiment
`20261101T121324Z_state_farm_45ee31db`, source revision
`7af1748e7a1f32dc7d12ea4c2a6801e6d094e27d`.

The full 10-epoch run completed successfully on a Tesla T4 (`returncode: 0`,
`fast: false`). Epoch 10 had the highest validation macro-F1 and was selected.
Test scoring remained disabled; there is no EfficientNet test accuracy yet.
No DMD inference or evaluation was performed.

## Comparison on the same validation drivers

All runs below use the same 2,712 images from the same three validation drivers.
Architecture comparison retains the original pretrained MobileNet training recipe;
the regularized MobileNet is a separate combined-recipe experiment.

| Model/run | Validation accuracy | Validation macro-F1 |
| --- | ---: | ---: |
| Pretrained MobileNetV3-Small | 66.04% | 63.15% |
| Regularized MobileNetV3-Small | 69.99% | 61.84% |
| Pretrained EfficientNet-B0 | **86.50%** | **81.18%** |

EfficientNet is the best completed validation candidate under the preselected
macro-F1 criterion: +18.04 percentage points over the original pretrained MobileNet.
Validation accuracy improved by 20.46 percentage points. These are single-run
measurements on three drivers, not confidence intervals or external generalization
results. The 85% aggregate validation accuracy goal is reached for this task;
per-class and final-test goals remain distinct.

| Validation class | F1 | Recall | Images |
| --- | ---: | ---: | ---: |
| Safe driving | 59.70% | 43.64% | 275 |
| Mobile phone use | 90.84% | 92.57% | 1,103 |
| Eating/drinking mapped class | 93.43% | 94.12% | 272 |
| Other-object mapped class | 91.56% | 90.75% | 789 |
| Passenger interaction | 70.39% | 85.35% | 273 |

All five class F1 scores improved over the original pretrained MobileNet. However,
safe-driving recall remains weak: 120/275 safe frames were correct, 96 were called
passenger interaction, and 58 phone use. The dataset mapping still does not supply
gaze training labels or full drowsiness events.

| Validation driver | Images | Accuracy |
| --- | ---: | ---: |
| p016 | 1,078 | 91.65% |
| p064 | 820 | 79.15% |
| p075 | 814 | 87.10% |

Per-driver figures were joined to the saved validation split in prediction order,
with all ground-truth labels checked. They show improved performance across these
drivers without establishing performance on arbitrary new drivers or vehicles.

## Verified local artifacts

Step 7's archive was present in the saved notebook and restored locally with
`python -m src.train.notebook_backup notebooks/driver_monitoring_training.ipynb`.

Local experiment folder (ignored by Git):
`results/experiments/20261101T121324Z_state_farm_45ee31db/`.

Restoration verified the archive SHA-256 and ZIP integrity. Additional checks passed:

- All 10 history entries are present, and the selected epoch/score equal the
  maximum validation macro-F1 in that history.
- Used splits exactly equal the fixed prepared reference; leakage checks passed.
- Manifest and split SHA-256 match run provenance. Every recorded source/config
  SHA-256 matches the local source used for the candidate.
- The 16,335,077-byte checkpoint loads through the demo's EfficientNet Classifier.
  A blank-frame inference returned a supported class and finite probability;
  this validates software compatibility, not real-video accuracy.
- Config and summary agree that test scoring is disabled; no `test.json` exists.

Embedded archive SHA-256:
`eefa349be23c9c4134188398d07b49108d8b7e56e966d48f10a8dbba1fea0f4f`.

Checkpoint SHA-256:
`79fb5a2821e023409d3be7bb1adb2c6d4fd60fdeca71e3a3ba1bf576588b7864`.

The model and full experiment artifacts are now independent of the Kaggle session.
Retain the earlier checkpoints and this notebook's saved attachment.

## Next decision

EfficientNet epoch 10 is the selected candidate based on validation. Freeze its
checkpoint, class mapping and preprocessing before final held-out evaluation.
That evaluation should load this exact checkpoint without retraining. Test results
must not be used to select another recipe. CPU latency and external DMD preparation
remain separate outstanding tasks; this result covers per-frame State Farm
distraction only.
