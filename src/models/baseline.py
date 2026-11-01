"""B2: plain per-frame CNN. No temporal stage and no implicit weight downloads."""

from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B2_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    efficientnet_b2,
    mobilenet_v3_small,
    resnet18,
)

SUPPORTED_MODELS = ("mobilenet_v3_small", "efficientnet_b0", "efficientnet_b2", "resnet18")


def build_baseline(num_classes=2, pretrained=False, model_name="mobilenet_v3_small"):
    if model_name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)
    elif model_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
    elif model_name == "efficientnet_b2":
        weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
        model = efficientnet_b2(weights=weights)
    elif model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unsupported classifier architecture: {model_name}")
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model
