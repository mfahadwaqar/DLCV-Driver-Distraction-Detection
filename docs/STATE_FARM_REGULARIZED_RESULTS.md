# Regularized State Farm candidate results

Completed on Kaggle on 2026-11-01. Experiment:
`20261101T080231Z_state_farm_17193824`; source revision
`c6c12c60b0d933953fdd34f1ba95c9aab10a48de`.

The run completed successfully (`returncode: 0`, `fast: false`) on a Tesla T4.
Early stopping ended training after 15 epochs. Epoch 9 was selected using the
highest validation macro-F1. Test scoring was disabled throughout this candidate
run; no new test or DMD result is available or claimed.

## Validation comparison

Both columns use the same 2,712 images from three validation drivers. The earlier
baseline reference is
`results/kaggle/20261029T074134312421Z/artifacts/state_farm_pretrained/val.json`.

| Validation measure | Baseline | Regularized candidate | Change in percentage points |
| --- | ---: | ---: | ---: |
| Accuracy | 66.04% | 69.99% | +3.95 |
| Macro-F1 | 63.15% | 61.84% | -1.31 |
| Safe-driving F1 | 56.20% | 33.42% | -22.78 |
| Phone-use F1 | 67.54% | 83.88% | +16.35 |
| Eating/drinking mapped-class F1 | 72.44% | 80.09% | +7.65 |
| Other-object mapped-class F1 | 76.48% | 72.30% | -4.19 |
| Passenger-interaction F1 | 43.08% | 39.51% | -3.57 |

The candidate improves aggregate validation accuracy and phone classification,
but does not improve the preselected validation macro-F1 criterion. Retain the
earlier baseline as the current reference under that criterion and preserve this
candidate for analysis. The 85% accuracy target remains unmet. One seed and three
validation drivers do not establish statistical significance or external performance.

Safe-driving recall fell from 65.09% to 23.64%. Of 275 safe-driving validation
frames, 142 were predicted as passenger interaction and 57 as the other-object
class. The next analysis should review these validation errors and class mappings
before another training recipe is selected. Do not select changes using test scores.

## Local backup and artifact verification

Step 7's model attachment was present in the saved local notebook. It was decoded,
SHA-256 checked, ZIP integrity checked, and restored with
`python -m src.train.notebook_backup notebooks/driver_monitoring_training.ipynb`.
No manual ZIP download was needed for this restoration.

Restored folder (ignored by Git):
`results/experiments/20261101T080231Z_state_farm_17193824/`.

The experiment includes `model/best.pt`, validation metrics/predictions, 15-epoch
history, exact used splits, effective configs, source/input provenance, environment,
completion status and the training log. It contains no new test metrics.

Verified locally:

- The checkpoint is 6,214,953 bytes and loads through the existing demo Classifier.
  Synthetic blank-frame inference returns a supported class and finite probability;
  this proves software compatibility, not real-video accuracy.
- `used_splits.json` exactly equals the fixed prepared reference assignments;
  subject leakage checks pass, and the manifest SHA-256 matches run provenance.
- The selected epoch and validation score agree with the maximum in the actual
  history; config and summary both confirm test scoring was disabled.

Embedded archive SHA-256:
`853084f2c7d4a95cbac47ebfe73334a31f1680a4a6f04898c534dc3f9acadc6a`.

Checkpoint SHA-256:
`1813f54bd35bda380759cff95aa4041101b9ddce28056b7e2ed823a12d8256ba`.

The checkpoint and outputs are now independent of the Kaggle session. Preserve
the local files and notebook attachment; raw/model data are not committed to Git.
