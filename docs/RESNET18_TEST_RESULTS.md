# Frozen ResNet-18 test results

Completed locally on 2026-11-03 at 11:32:45 UTC. The saved testing notebook and
all generated artifacts were checked after completion. No training, checkpoint
reselection, threshold tuning or second inference run was performed by this check.

## Main result

The exact validation-selected epoch-10 ResNet-18 checkpoint correctly classified
**3,717 / 4,831 test frames** from five drivers excluded from its training and
validation sets. Test accuracy is **76.94%**, and macro-F1 is **69.16%**.

| Measurement | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| ResNet-18 validation, 3 drivers | 87.68% | 85.12% |
| ResNet-18 test, 5 different drivers | **76.94%** | **69.16%** |
| Earlier pretrained MobileNet test, same 5 test drivers | 64.04% | 58.48% |

ResNet-18 improves over the earlier pretrained MobileNet test result by 12.90
percentage points in accuracy and 10.68 points in macro-F1. Historical MobileNet
sample identities, driver memberships and labels were checked against all current
splits, allowing this comparison despite remapped paths/namespaces. It does not
establish how EfficientNet-B0 or B2 would score on test; those models were not
scored here or selected using test results.

Accuracy is 10.74 points below the ResNet validation result; macro-F1 is 15.96
points below validation. Performance varies across drivers and classes. This is
evidence of limited generalization to the five test drivers, not a failed
notebook execution. It does not isolate the cause to architecture, labels,
training settings or visual differences. The 90% accuracy goal is not reached.

## Per-class results

Recall is the fraction of actual frames of that class detected correctly.
Precision is the fraction of predictions for that class that were correct.

| Class | Precision | Recall | F1 | Correct / actual frames |
| --- | ---: | ---: | ---: | ---: |
| Safe driving | 67.45% | 91.78% | 77.76% | 458 / 499 |
| Mobile phone use | 96.52% | 70.40% | 81.41% | 1,441 / 2,047 |
| Eating/drinking mapped class | 83.13% | 92.10% | 87.39% | 478 / 519 |
| Other-object mapped class | 65.08% | 94.54% | 77.09% | 1,282 / 1,356 |
| Passenger interaction | 50.88% | 14.15% | 22.14% | 58 / 410 |

Passenger interaction is the major weakness: 182 frames were predicted as other
object and 160 as safe driving. Phone recall is also incomplete: 463 phone frames
were predicted as other object. High safe-driving recall does not mean high
safe-driving precision; many passenger frames were mistakenly called safe.

| Test driver | Correct / frames | Accuracy |
| --- | ---: | ---: |
| p002 | 584 / 725 | 80.55% |
| p015 | 540 / 875 | 61.71% |
| p024 | 842 / 1,226 | 68.68% |
| p026 | 1,089 / 1,196 | 91.05% |
| p061 | 662 / 809 | 81.83% |

These are five drivers with correlated frames, not thousands of independent
participants. No confidence intervals or arbitrary-driver guarantees are claimed.

## Verified artifacts

Reports and predictions:
`results/testing/state_farm_resnet18_frozen/`.
Report archive:
`results/testing/state_farm_resnet18_frozen_reports.zip`.
Executed notebook backup:
`results/notebook-backups/completed-resnet18-test-631f3494b57b.ipynb`.

Verification passed:

- Frozen checkpoint/configuration, manifest/split hashes and original subject
  separation remain valid. Evaluation provenance matches the evaluator source.
- Exactly 4,831 predictions align by path, driver and true label with the fixed
  test rows; class probabilities are finite, normalized and match argmax labels.
- Predictions reproduce aggregate accuracy, macro-F1, confusion matrix and all
  five per-driver reports.
- The seven report files in the ZIP are byte-identical to the local reports;
  ZIP CRC verification passes.
- The completed notebook is separately preserved, and model weights are unchanged.

Checkpoint SHA-256:
`c4e21db23cd3250fc98dec9f8ea617c162d70273cacc3938ee43fca4ceb6dfe8`.

Local CPU batch inference took 200.71 seconds (about 3 minutes 21 seconds).
This measures the notebook inference workload, not live-video latency or full
system throughput.

## Scope and next work

This completes fixed-checkpoint State Farm distraction testing. Keep the model
and recipe fixed rather than changing them to improve this known test result.
State Farm test had already been scored for earlier MobileNet baselines, so it
is an existing fixed benchmark, not a never-inspected external dataset.

Next, integrate the selected distraction checkpoint with the existing eye-state
checkpoint in the video demo and measure video behavior/CPU latency. Prepare
camera/timestamp/label alignment before DMD external evaluation. NITYMED still
needs reviewed identities and event intervals for yawning/microsleep supervision.
This report does not score eye closure, yawning, nodding, gaze, video events or
DMD, and does not complete the full proposed monitoring system.
