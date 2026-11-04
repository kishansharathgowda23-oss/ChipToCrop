"""Optional detector and segmentation architecture factories from the blueprint."""
from __future__ import annotations


def build_yolov8n(weights: str = "yolov8n.pt"):
    """Load the planned nano detector. Dataset conversion/training is separate."""
    if weights != "yolov8n.pt":
        raise ValueError("This factory currently supports only the planned yolov8n.pt checkpoint.")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install the optional `ultralytics` package to use YOLOv8n.") from exc
    return YOLO(weights)


def build_deeplabv3(num_classes: int, pretrained_backbone: bool = True):
    """DeepLabV3-ResNet50 with a project-sized output head."""
    import torch.nn as nn
    from torchvision import models

    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")
    backbone_weights = models.ResNet50_Weights.DEFAULT if pretrained_backbone else None
    model = models.segmentation.deeplabv3_resnet50(weights=None, weights_backbone=backbone_weights)
    model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
    if model.aux_classifier is not None:
        model.aux_classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model


def build_unet(num_classes: int, base_channels: int = 32):
    """Small U-Net for pixel masks; training requires reviewed mask annotations."""
    import torch.nn as nn
    import torch.nn.functional as F

    if num_classes < 2 or base_channels < 4:
        raise ValueError("num_classes must be >=2 and base_channels must be >=4.")

    class DoubleConv(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            )

        def forward(self, value):
            return self.layers(value)

    class UNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            c = base_channels
            self.enc1 = DoubleConv(3, c)
            self.enc2 = DoubleConv(c, c * 2)
            self.enc3 = DoubleConv(c * 2, c * 4)
            self.bottleneck = DoubleConv(c * 4, c * 8)
            self.pool = nn.MaxPool2d(2)
            self.dec3 = DoubleConv(c * 8 + c * 4, c * 4)
            self.dec2 = DoubleConv(c * 4 + c * 2, c * 2)
            self.dec1 = DoubleConv(c * 2 + c, c)
            self.output = nn.Conv2d(c, num_classes, kernel_size=1)

        def forward(self, value):
            e1 = self.enc1(value)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            b = self.bottleneck(self.pool(e3))
            d3 = self.dec3(torch_cat_upsample(b, e3, F))
            d2 = self.dec2(torch_cat_upsample(d3, e2, F))
            d1 = self.dec1(torch_cat_upsample(d2, e1, F))
            return self.output(d1)

    def torch_cat_upsample(value, skip, functional):
        value = functional.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        import torch
        return torch.cat((skip, value), dim=1)

    return UNet()
