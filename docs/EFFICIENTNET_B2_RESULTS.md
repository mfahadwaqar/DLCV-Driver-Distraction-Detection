# EfficientNet-B2 validation results

Completed and verified locally on 2026-11-03. Experiment:
`20261103T085212Z_state_farm_efficientnet_b2_73e0ed78`.
Source revision: `71bfe9113d3aac4cc98558a865263ef6d42caf69`.

All ten epochs completed on a Tesla T4 with return code zero. Epoch 6 was selected
by the highest validation macro-F1, not the final epoch or test performance.
The run used pretrained EfficientNet-B2, 288-pixel inputs, batch size 16 and Adam
at 0.0001. Only the State Farm training split (14,881 images, 18 drivers) trained
this model. RT-BENE, NITYMED and DMD were not included in its training.

## Same validation set comparison

All candidates below were measured on the same 2,712 frames from three validation
drivers. B2 changes architecture, input resolution and batch size together.

| Model | Validation accuracy | Validation macro-F1 |
| --- | ---: | ---: |
| Pretrained MobileNetV3-Small | 66.04% | 63.15% |
| Regularized MobileNetV3-Small | 69.99% | 61.84% |
| EfficientNet-B0 | **86.50%** | **81.18%** |
| EfficientNet-B2 | 84.85% | 78.06% |

B2 is lower than B0 by 1.66 percentage points in accuracy and 3.12 points in
macro-F1. B0 remains the selected candidate. Neither result is final-test
accuracy, and neither establishes full drowsiness performance or a 90% result.

| Validation class | B0 F1 | B2 F1 | B0 recall | B2 recall |
| --- | ---: | ---: | ---: | ---: |
| Safe driving | 59.70% | 71.40% | 43.64% | 64.00% |
| Mobile phone use | 90.84% | 95.96% | 92.57% | 93.56% |
| Eating/drinking mapped class | 93.43% | 92.77% | 94.12% | 96.69% |
| Other-object mapped class | 91.56% | 81.56% | 90.75% | 91.63% |
| Passenger interaction | 70.39% | 48.64% | 85.35% | 39.19% |

B2 correctly recognized more safe frames (176/275 versus 120/275), but only
107/273 passenger frames; 129 passenger frames were classified as other object.
That confusion explains much of the lower macro-F1. Training loss fell from
0.2361 to 0.0082 while validation macro-F1 fluctuated and peaked at epoch 6.
This is consistent with limited generalization despite fitting the training
frames; it does not establish a single causal explanation. A larger backbone
did not improve this single-run comparison.

| Validation driver | Correct / images | B2 accuracy |
| --- | ---: | ---: |
| p016 | 917 / 1,078 | 85.06% |
| p064 | 709 / 820 | 86.46% |
| p075 | 675 / 814 | 82.92% |

## Recovery and verification

The saved notebook contained the complete Step 7 backup. It was restored to
`results/experiments/20261103T085212Z_state_farm_efficientnet_b2_73e0ed78/`.
The executed notebook was also preserved at
`results/notebook-backups/completed-efficientnet-b2-490435abd921.ipynb`.
Both are ignored by Git; model files remain local.

Checks passed:

- Embedded archive SHA-256 and ZIP CRC verified during safe restoration.
- Ten history entries; selected epoch and macro-F1 equal the history maximum.
- Used splits exactly match the fixed reference, with no subject leakage.
- Manifest/split hashes and all 35 recorded source/config hashes match locally.
- All 2,712 validation prediction labels align with the saved split; predictions
  reproduce the stored confusion matrix, accuracy and macro-F1.
- The actual 31,260,161-byte checkpoint loads through the demo's B2 classifier.
  Blank-frame inference returns a supported class and finite probability. This
  checks software compatibility, not real-video accuracy.
- Test scoring is disabled in both configuration and summary; no test artifacts
  were produced. DMD remains excluded from training, tuning and scoring.

Archive SHA-256:
`295c4ec4515f99d37a745285c01c8238cc947341572b7e3f7d8f897821dbc420`.

Checkpoint SHA-256:
`755a96416f837a365f5fed8a28cafb73d7a2d55a2a174c32766a56db21c51de5`.

## Next comparison

Run the already prepared ResNet-18 candidate using the same uploaded code and
data. Change only the notebook's model selector to `"resnet18"`; inspect with
training disabled, then enable training for one full run. Save the final backup.
Keep B0 unless the remaining candidate improves validation macro-F1. Do not
select candidates using held-out test scores. See [the run guide](MODEL_COMPARISON.md).
