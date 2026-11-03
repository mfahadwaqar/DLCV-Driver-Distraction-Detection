# ResNet-18 validation results

Completed and verified locally on 2026-11-03. Experiment:
`20261103T103006Z_state_farm_resnet18_d497ae72`.
Source revision: `71bfe9113d3aac4cc98558a865263ef6d42caf69`.

All ten epochs completed on a Tesla T4 with return code zero. Epoch 10 had the
highest validation macro-F1 and was selected. This run used ImageNet pretrained
ResNet-18, 224-pixel inputs, batch size 32 and Adam at 0.0001. Training used
14,881 State Farm frames from 18 drivers. The other datasets did not train this
model; both prepared datasets showing READY is an input check, not joint training.

## Completed comparison

Every candidate below uses the same 2,712 validation frames from three drivers.

| Model | Validation accuracy | Validation macro-F1 |
| --- | ---: | ---: |
| Pretrained MobileNetV3-Small | 66.04% | 63.15% |
| Regularized MobileNetV3-Small | 69.99% | 61.84% |
| EfficientNet-B0 | 86.50% | 81.18% |
| EfficientNet-B2 | 84.85% | 78.06% |
| **ResNet-18** | **87.68%** | **85.12%** |

ResNet-18 is now the selected distraction candidate under the preselected
validation macro-F1 criterion. Relative to B0, accuracy improved by 1.18 percentage
points and macro-F1 by 3.94 points. The 90% aggregate accuracy goal has not been
reached. These are single-run validation measurements, not final-test scores,
uncertainty estimates or full driver-drowsiness results.

| Validation class | F1 | Recall | Images |
| --- | ---: | ---: | ---: |
| Safe driving | 86.59% | 82.18% | 275 |
| Mobile phone use | 92.61% | 90.93% | 1,103 |
| Eating/drinking mapped class | 95.10% | 96.32% | 272 |
| Other-object mapped class | 85.07% | 93.54% | 789 |
| Passenger interaction | 66.22% | 54.58% | 273 |

Safe-driving recall improved substantially over B0 (43.64% to 82.18%). Passenger
interaction remains weak: 149/273 frames were correct and 94 were called other
object. A high overall score therefore does not mean every class reaches 90%.

| Validation driver | Correct / images | Accuracy |
| --- | ---: | ---: |
| p016 | 1,044 / 1,078 | 96.85% |
| p064 | 704 / 820 | 85.85% |
| p075 | 630 / 814 | 77.40% |

These differences show uneven performance across the three validation drivers.
ResNet-18 does not outperform B0 on every driver or class. Repeated comparisons
on this validation set also limit conclusions about arbitrary unseen drivers.

## Verified local recovery

The complete Step 7 archive was restored from the saved local notebook to:
`results/experiments/20261103T103006Z_state_farm_resnet18_d497ae72/`.
The executed notebook is separately preserved at:
`results/notebook-backups/completed-resnet18-c75d968f5393.ipynb`.
These artifacts are ignored by Git and no longer depend on the remote session.

Checks passed:

- Safe restoration verified the embedded archive SHA-256 and ZIP CRC.
- Ten history entries; selected epoch 10 and score match the history maximum.
- Used splits exactly match the saved reference, with no subject leakage.
- Manifest/split hashes and all 35 recorded source/config hashes match locally.
- All 2,712 prediction labels align with the saved validation rows. Predictions
  reproduce the confusion matrix, accuracy and macro-F1.
- The actual 44,788,683-byte checkpoint loads through the demo's ResNet classifier.
  Blank-frame inference returns a supported class and finite probability. This
  proves software compatibility, not real-video accuracy.
- Test scoring is disabled in configuration and summary, and there are no test
  artifacts. No DMD model scoring was performed.

Archive SHA-256:
`c9f1fd34a190b81192d47d66dd15426a49bab6069e8daeac7ae8255d37bfe514`.

Checkpoint SHA-256:
`c4e21db23cd3250fc98dec9f8ea617c162d70273cacc3938ee43fca4ceb6dfe8`.

## Next stage

The planned two-candidate comparison is complete. Freeze this checkpoint,
224-pixel inference transform and class mapping before evaluating it on the
existing 4,831-image, five-driver State Farm test split. Load the exact checkpoint
without retraining. That split was previously scored for the MobileNet baselines,
so it is a fixed benchmark, not a dataset that has never been inspected. Do not
use its scores to choose further recipes or checkpoints.

Then integrate the selected distraction checkpoint with the existing RT-BENE
eye-state checkpoint and measure actual video behavior and CPU latency. DMD
requires camera/timestamp/label alignment before external evaluation. NITYMED
still needs reviewed participant identities and event intervals before event
training. These are separate remaining tasks; current image classification does
not supply yawning, nodding or gaze performance. No additional downloads are
needed to begin checkpoint evaluation and demo integration.
