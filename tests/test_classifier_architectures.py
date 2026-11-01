import pytest
from torch import nn

import src.models.baseline as factory


@pytest.mark.parametrize("architecture", ["mobilenet_v3_small", "efficientnet_b0"])
@pytest.mark.parametrize("pretrained", [False, True])
def test_architecture_requests_expected_weights_and_replaces_head(
    monkeypatch, architecture, pretrained
):
    calls = []

    def constructor(*, weights):
        calls.append(weights)
        model = nn.Module()
        model.classifier = nn.Sequential(nn.Linear(7, 1000))
        return model

    monkeypatch.setattr(factory, architecture, constructor)
    model = factory.build_baseline(5, pretrained=pretrained, model_name=architecture)
    enum = (
        factory.EfficientNet_B0_Weights
        if architecture == "efficientnet_b0"
        else factory.MobileNet_V3_Small_Weights
    )
    assert calls == [enum.DEFAULT if pretrained else None]
    assert model.classifier[-1].in_features == 7
    assert model.classifier[-1].out_features == 5


def test_unknown_architecture_cannot_silently_train_mobile_baseline():
    with pytest.raises(ValueError, match="Unsupported classifier architecture"):
        factory.build_baseline(model_name="typo")
