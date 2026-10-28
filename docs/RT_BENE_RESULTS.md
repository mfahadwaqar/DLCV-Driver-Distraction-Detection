# RT-BENE eye-state baseline: first full Kaggle run

The saved private Kaggle run completed on 2026-10-28. It used source commit
`bc0bb0e6b7687c63db7fff6aac9d11044f451928`, Python 3.13.15, PyTorch
2.11.0+cu128 and a Tesla T4. The model was MobileNetV3-Small trained from random
initialization for five epochs with training-split class weights. The best
validation checkpoint came from epoch 3. The downloaded outputs are in the
ignored local folder `results/kaggle/20261028T075933392791Z/artifacts/phase1/`.

RT-BENE here labels individual eye crops as open or closed/blinking. These labels
do not distinguish a blink from sustained drowsiness or sleep. No full-face,
distraction, yawning, nodding, event-level or in-truck performance is measured.

| Measure | Validation | Held-out test |
| --- | ---: | ---: |
| Subjects | 2 | 3 |
| Eye images | 9,299 | 17,005 |
| Macro F1 | 0.881 | 0.900 |
| Closed-eye precision | 0.753 | 0.890 |
| Closed-eye recall | 0.779 | 0.747 |
| Closed-eye F1 | 0.766 | 0.812 |

Training used 81,046 images from 11 subjects: 77,782 open and 3,264
closed/blinking. The saved subject assignments were checked for overlap. The
test confusion matrix, with rows as truth and columns as prediction, is:

| True / predicted | Open | Closed/blinking |
| --- | ---: | ---: |
| Open | 15,781 | 103 |
| Closed/blinking | 284 | 837 |

The model missed 284 of 1,121 closed/blinking test images. The test split has a
higher closed-eye share than validation, and this is one run with one split and
seed. Avoid presenting 97.7% accuracy as the primary result because open eyes
dominate. The relevant result is the class-specific recall and macro F1.

## Error analysis by held-out subject

The saved prediction rows were matched to the saved test split in order, with
labels checked for agreement. The analysis can be reproduced with
`scripts/analyze_rt_bene_errors.py`; its per-subject CSV and ranked false
negative/positive frame lists are in the ignored local `error_analysis/` folder
beside the downloaded run artifacts.

| Subject | Closed images | Missed closed | Closed recall | Open false alarms |
| --- | ---: | ---: | ---: | ---: |
| s000 | 931 | 249 | 73.3% | 6 |
| s003 | 167 | 18 | 89.2% | 81 |
| s012 | 23 | 17 | 26.1% | 16 |

s000 contributes 249 of the 284 missed closed-eye images. s012 has the lowest
closed-eye recall, but only 23 positive samples, so the estimate is less stable.
s003 has most false alarms (81 of 103). These numbers flag subjects to inspect;
they do not explain why errors occur. The prepared samples have no glasses or
lighting condition labels, so this run cannot support those subgroup claims.

Artifacts include `best.pt`, `summary.json`, `history.json`, split manifests,
per-split predictions, confusion matrices, configuration and environment. The
downloaded checkpoint loaded locally, all 17,005 test predictions were present,
and saved train/validation/test subject sets were disjoint.

Next: inspect errors by subject and lighting/glasses where metadata permits,
then build and evaluate the distraction baseline on a separate dataset. Request
an unseen final-presentation test sample from the professor now that the project
has its first real result.
