import pytest
from torch import nn

import src.models.baseline as factory


@pytest.mark.parametrize("architecture", factory.SUPPORTED_MODELS)
@pytest.mark.parametrize("pretrained", [False, True])
def test_architecture_requests_expected_weights_and_replaces_head(
    monkeypatch, architecture, pretrained
):
    calls = []

    def constructor(*, weights):
        calls.append(weights)
        model = nn.Module()
        model.classifier = nn.Sequential(nn.Linear(7, 1000))
        model.fc = nn.Linear(7, 1000)
        return model

    monkeypatch.setattr(factory, architecture, constructor)
    model = factory.build_baseline(5, pretrained=pretrained, model_name=architecture)
    enum = {
        "mobilenet_v3_small": factory.MobileNet_V3_Small_Weights,
        "efficientnet_b0": factory.EfficientNet_B0_Weights,
        "efficientnet_b2": factory.EfficientNet_B2_Weights,
        "resnet18": factory.ResNet18_Weights,
    }[architecture]
    assert calls == [enum.DEFAULT if pretrained else None]
    head = model.fc if architecture == "resnet18" else model.classifier[-1]
    assert head.in_features == 7
    assert head.out_features == 5


def test_unknown_architecture_cannot_silently_train_mobile_baseline():
    with pytest.raises(ValueError, match="Unsupported classifier architecture"):
        factory.build_baseline(model_name="typo")
