import json

import pytest
import torch
from PIL import Image
from torchvision import transforms

from src.data.mrl import ImageDataset, index_mrl
from src.data.splits import saved_splits
from src.demo.run import Classifier
from src.models.baseline import SUPPORTED_MODELS
from src.train.phase1 import image_transform, train


def test_driver_augmentation_is_train_only_and_preserves_output_shape():
    train_transform = image_transform(64, train=True, augmentation="driver")
    assert any(isinstance(op, transforms.ColorJitter) for op in train_transform.transforms)
    assert not any(
        isinstance(op, transforms.RandomHorizontalFlip) for op in train_transform.transforms
    )
    evaluation = image_transform(64, augmentation="driver")
    assert not any(isinstance(op, transforms.ColorJitter) for op in evaluation.transforms)
    image = Image.new("RGB", (80, 60), (100, 50, 30))
    assert train_transform(image).shape == (3, 64, 64)
    assert torch.equal(evaluation(image), image_transform(64)(image))
    with pytest.raises(ValueError, match="Unknown augmentation"):
        image_transform(64, augmentation="typo")


@pytest.mark.parametrize("architecture", SUPPORTED_MODELS)
def test_validation_only_recipe_never_opens_test_images_and_checkpoint_loads(
    tmp_path, monkeypatch, architecture
):
    data = tmp_path / "data"
    data.mkdir()
    for subject in range(1, 7):
        for eye in (0, 1):
            Image.new("RGB", (64, 64), (eye * 120, 40, 80)).save(
                data / f"s{subject:04d}_00001_0_0_{eye}_0_1_01.png"
            )
    split_path = tmp_path / "splits.json"
    parts = saved_splits(index_mrl(data), split_path)
    forbidden = {r["path"] for r in parts["test"]}
    original = ImageDataset.__getitem__

    def guarded(self, index):
        assert self.rows[index]["path"] not in forbidden, "Test image accessed during selection"
        return original(self, index)

    monkeypatch.setattr(ImageDataset, "__getitem__", guarded)
    config = dict(
        seed=42,
        model=architecture,
        split_fractions=[0.7, 0.15, 0.15],
        fast_samples_per_split=8,
        image_size=64,
        batch_size=4,
        num_workers=0,
        pretrained=False,
        learning_rate=0.001,
        fast_epochs=1,
        epochs=1,
        augmentation="driver",
        optimizer="adamw",
        weight_decay=0.01,
        scheduler="cosine",
        early_stopping_patience=6,
        evaluate_test=False,
    )
    output = tmp_path / "model"
    summary = train(config, data, output, split_path)
    assert summary["test_evaluated"] is False
    assert summary["model"] == architecture
    assert set(summary["macro_f1"]) == {"val"}
    assert not (output / "test.json").exists()
    assert not (output / "test_predictions.csv").exists()
    assert summary["best_epoch"] == 1
    assert len(json.loads((output / "history.json").read_text())) == 1
    classifier = Classifier(output, "label_drowsiness")
    head = classifier.model.fc if architecture == "resnet18" else classifier.model.classifier[-1]
    assert head.out_features == 2
