# State Farm results

The completed pretrained MobileNetV3-Small experiment achieved test macro F1 of **0.584755**, compared with **0.315119** for the random-initialization baseline, on exactly the same held-out drivers.

Both runs performed full training on Kaggle with a Tesla T4 (`fast: false`). Scores below come from saved evaluation JSON, not notebook completion status.

## Measured comparison

| Metric | Baseline | Pretrained | Absolute change |
| --- | ---: | ---: | ---: |
| val macro F1 | 0.427695 | 0.631478 | +0.203783 |
| val accuracy | 0.542773 | 0.660398 | +0.117625 |
| test macro F1 | 0.315119 | 0.584755 | +0.269635 |
| test accuracy | 0.443593 | 0.640447 | +0.196854 |

Macro F1 gives each of the five supported classes equal weight. Accuracy is the fraction of correctly classified frames. These are single-run measurements without confidence intervals.

| Test class | Baseline F1 | Pretrained F1 |
| --- | ---: | ---: |
| Safe driving | 0.099707 | 0.579216 |
| Mobile phone use | 0.496936 | 0.633508 |
| Eating or drinking | 0.435005 | 0.763731 |
| Other object | 0.525684 | 0.713275 |
| Talking to passenger | 0.018265 | 0.234043 |

## Training and checkpoint selection

Both experiments used 224-pixel inputs, batch size 32, seed 42, the same five project class IDs `[0, 1, 2, 3, 5]`, and the same training class weights. The baseline used random initialization, 5 epochs, and learning rate 0.001. The pretrained experiment used ImageNet initialization, 10 epochs, and learning rate 0.0001. Consequently, this comparison measures the combined training recipe change; it does not isolate pretraining alone.

The saved histories identify baseline epoch **4** as best (validation macro F1 0.427695), and pretrained epoch **8** as best (validation macro F1 0.631478). The training code selects `best.pt` using validation macro F1 and reloads it before final validation/test evaluation. The final epoch is not the selected checkpoint.

Baseline source commit: `d5d8a4851c820b2d934c8ad222c7ed57b37079a7`. Pretrained source commit: `60a40fc4e39dbb12ffaace2a1f14734fceb02fc7`.

## Verification and artifacts

The pretrained `used_splits.json` exactly equals both the baseline `used_splits.json` and the saved baseline split document's `splits` field, including row order and metadata. The saved fingerprint was recomputed successfully and subject leakage checks passed.

| Split | Frames | Drivers |
| --- | ---: | ---: |
| Train | 14,881 | 18 |
| Validation | 2,712 | 3 |
| Test | 4,831 | 5 |

The downloaded pretrained `best.pt` (6,214,953 bytes) loaded successfully through `src.demo.run.Classifier` with target `label_distraction`. CPU inference on a synthetic RGB frame returned a valid class and finite probability. This verifies checkpoint/demo compatibility, not real-video accuracy or temporal alert performance.

Local output directories (ignored by Git):

- Baseline: `results/kaggle/20261029T065436396085Z/artifacts/state_farm/`.
- Pretrained Kaggle version 4: `results/kaggle/20261029T074134312421Z/artifacts/state_farm_pretrained/`.
- Saved reference split: `results/kaggle/20261029T065436396085Z/artifacts/state_farm_splits.json`.

Each experiment directory contains `summary.json`, `val.json`, `test.json`, `history.json`, `config.yaml`, `best.pt`, confusion matrices, predictions, and `used_splits.json`.

## Limits

- These are per-frame State Farm distraction results. They do not measure drowsiness, gaze-off-road recognition, event-level alerts, trucking conditions, or deployment safety.
- Test talking-to-passenger F1 remains low (0.234043); phone-use recall is 0.472887 despite precision of 0.959366. Aggregate improvement does not imply uniformly reliable detection.
- Only three validation drivers and five test drivers were evaluated. No repeated seeds, external dataset validation, or in-subject comparison was performed.
- No model export or real-video demo was validated by the checkpoint-loading check.

## Readable notebook publication

The updated generator uses visible setup/data/split/training/results cells, a public GitHub checkout pinned to the committed SHA, and the exact baseline rows as plain Python data. It contains no encoded/compressed payloads or hidden source cells. All nine workflow tests and Ruff checks passed; full saved split restoration was verified locally.

The installed Kaggle CLI supports `kernels push -p FOLDER --no-run`: it saves a new notebook version without executing any cell. This can publish the readable source after committing/pushing the pinned code and preparing the notebook. It does not create fresh training outputs; measured results remain those of completed version 4. No publication or rerun was performed during this verification.
