"""Model factory for the planned image-classification benchmark pool."""
from __future__ import annotations

CLASSIFIER_MODELS = (
    "mobilenet_v3_large",
    "efficientnet_b0",
    "resnet50",
    "densenet121",
    "convnext_tiny",
)
PRIMARY_MODEL = "mobilenet_v3_large"
COMPARISON_MODELS = ("efficientnet_b0", "resnet50")
OPTIONAL_MODELS = ("densenet121", "convnext_tiny")


def build_classifier(name: str, num_classes: int, pretrained: bool = True):
    import torch.nn as nn
    from torchvision import models

    if name not in CLASSIFIER_MODELS:
        raise ValueError(f"Unsupported classifier {name!r}; choose from {CLASSIFIER_MODELS}.")

    weight_map = {
        "mobilenet_v3_large": models.MobileNet_V3_Large_Weights.DEFAULT,
        "efficientnet_b0": models.EfficientNet_B0_Weights.DEFAULT,
        "resnet50": models.ResNet50_Weights.DEFAULT,
        "densenet121": models.DenseNet121_Weights.DEFAULT,
        "convnext_tiny": models.ConvNeXt_Tiny_Weights.DEFAULT,
    }
    weights = weight_map[name] if pretrained else None

    if name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif name == "resnet50":
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "densenet121":
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model
