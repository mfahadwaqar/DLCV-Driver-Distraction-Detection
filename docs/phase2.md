# State Farm distraction baseline

Implementation is ready for a first Kaggle smoke run. No distraction training
result is claimed yet. RT-BENE checkpoint and metrics remain the eye-state baseline.

The adapter reads the official `driver_imgs_list.csv` and only the labeled
`imgs/train/c*/` images (also supports `train/c*/`). It retains driver IDs,
rejects missing/duplicate/invalid files, and excludes the unlabeled competition
test images. The existing persisted subject split implementation creates roughly
70/15/15 train/validation/test splits by driver, then selects the checkpoint using
validation macro F1. The project test split is held out until final evaluation.

| Source classes | Project class ID | Project label |
| --- | ---: | --- |
| c0 | 0 | Safe driving |
| c1, c2, c3, c4 | 1 | Mobile phone use |
| c6 | 2 | Eating or drinking (drinking examples only) |
| c5, c7, c8 | 3 | Other object |
| none | 4 | Gaze off road: unsupported |
| c9 | 5 | Talking to passenger |

The source classes describe radio use, reaching behind and hair/makeup; grouping
these under Other object is the project mapping, not an original source label.
State Farm cannot establish gaze-off-road recognition or eating performance.
Official source: https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/data

`configs/state_farm.yaml` trains a five-output MobileNetV3-Small from random
initialization at 224 pixels for five epochs. Images preserve full driver context.
Training uses class weights from the training split. Reports retain project class
IDs `[0, 1, 2, 3, 5]` and explicitly list gaze as unsupported. This is a separate
single-task baseline, before the shared model.

After committing the source, run the small integration check on Kaggle:

```powershell
.\.venv\Scripts\python.exe scripts/kaggle_workflow.py submit --task state-farm --device cuda --fast
```

The generated private notebook attaches the official competition input, indexes
it into a temporary manifest and saves only outputs under
`/kaggle/working/artifacts/state_farm`. Internet is disabled for this task.
After checking the smoke outputs, repeat without `--fast` for the full baseline.
The existing saved notebook receives a new version; previous saved versions and
downloaded RT-BENE artifacts remain available. Use the workflow `status` and
`fetch-results --version N` commands to inspect and preserve each version.

Local verification uses artificial images only: mapping, identity preservation,
unsupported gaze handling, end-to-end distraction artifact creation, existing
eye-state regression checks and competition-input packaging. It does not measure
distraction accuracy.
