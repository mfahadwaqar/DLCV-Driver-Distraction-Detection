# Proposed State Farm transfer-learning experiment

The random-initialization run's validation macro F1 peaked at 0.428 in epoch 4
while training loss fell to 0.102 by epoch 5. This indicates poor validation
generalization despite fitting the training drivers. It motivates a pretrained
backbone experiment; it does not establish its cause or guarantee improvement.

`configs/state_farm_pretrained.yaml` changes only initialization, learning rate
and epoch budget: ImageNet MobileNetV3-Small weights, learning rate 0.0001 and
10 epochs. The seed, class map, input size, augmentation, batch size, class-weight
calculation and driver assignment remain the same. The full network is fine-tuned.
This is a proposed experiment, not a measured result.

Use the baseline's exact persisted `state_farm_splits.json` as `--splits`, retaining
its data fingerprint, seed and fractions. The file is in the downloaded run's
`artifacts/` directory, one level above `state_farm/`. The trainer already verifies
these fields and rejects mismatches. Do not recreate or tune split assignments.
Choose the checkpoint by validation macro F1 only. Compare the final checkpoint
to the baseline using the same saved test assignment, without changing the
threshold or selecting a model using test results.

The existing `build_baseline(pretrained=True)` loads Torchvision's
`MobileNet_V3_Small_Weights.DEFAULT` (ImageNet V1). If weights are not cached, it
downloads the file from PyTorch's official model host. The State Farm saved-run
workflow currently disables internet; the reviewed submission must enable
internet for this experiment or attach an authorized weights file/cache.
Do not silently fall back to random initialization on a download failure.

Within Kaggle, after the normal State Farm manifest setup, the training command is:

```text
python -m src.train.phase1 --config configs/state_farm_pretrained.yaml --data-root /kaggle/input --manifest /kaggle/temp/state-farm/manifest.csv --splits PATH_TO_BASELINE_STATE_FARM_SPLITS_JSON --output /kaggle/working/artifacts/state_farm_pretrained --device cuda
```

The baseline split file must be made available in the new run. The new output
folder preserves the original baseline artifacts. No experiment has been
submitted as part of preparing this configuration.
