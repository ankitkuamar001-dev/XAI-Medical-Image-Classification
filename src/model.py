"""
============================================================
src/model.py
CNN Model Architecture
  - CustomCNN (built from scratch)
  - ResNet50  (pretrained transfer learning)  ← default
  - VGG16
  - EfficientNet-B0
  - DenseNet-121
============================================================

Architecture Decision Rationale
--------------------------------
ResNet50 is our default because:
  1. Skip connections solve vanishing gradients (deep network stability).
  2. Pretrained on ImageNet → rich low-level feature detectors (edges,
     textures) that transfer well to medical images.
  3. Layer4 (last conv block) produces 7×7 activation maps that are
     ideal for Grad-CAM heatmaps (interpretable spatial resolution).
  4. Well-studied in medical imaging literature for chest X-ray tasks.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from src.utils import get_logger

logger = get_logger("model")


# ================================================================== #
# 1. Custom CNN (from scratch)
# ================================================================== #
class CustomCNN(nn.Module):
    """
    4-block custom CNN for binary medical image classification.

    Architecture:
        Input (3 × 224 × 224)
        → Conv Block 1  (32  filters)
        → Conv Block 2  (64  filters)
        → Conv Block 3  (128 filters)
        → Conv Block 4  (256 filters)
        → AdaptiveAvgPool → Flatten
        → FC (512) → Dropout → FC (num_classes)

    Each Conv Block:
        Conv2d → BatchNorm → ReLU → MaxPool

    Design Choices:
        • BatchNorm: accelerates training by normalizing activations.
        • MaxPool: spatial downsampling, translational invariance.
        • Dropout (p=0.5): prevents co-adaptation of neurons (reduces overfit).
        • AdaptiveAvgPool: output is always 1×1 regardless of input size.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()

        def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            )

        self.features = nn.Sequential(
            conv_block(3,   32),    # 224 → 112
            conv_block(32,  64),    # 112 → 56
            conv_block(64,  128),   # 56  → 28
            conv_block(128, 256),   # 28  → 14
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))   # 14 → 1

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

    def get_gradcam_layer(self):
        """Return last conv block for Grad-CAM."""
        return self.features[-1]


# ================================================================== #
# 2. ResNet-50 (Transfer Learning) — DEFAULT
# ================================================================== #
class ResNet50Classifier(nn.Module):
    """
    ResNet-50 fine-tuned for medical image classification.

    Modifications:
        • Replace the final FC layer (fc) with a custom head:
          [Linear(2048 → 512) → ReLU → Dropout → Linear(512 → num_classes)]
        • Optionally freeze backbone layers to train only the head
          (useful when dataset is small).

    Grad-CAM target: model.backbone.layer4
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.5,
    ):
        super().__init__()

        # Load pretrained ResNet50
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen — training head only.")

        # Replace the classification head
        in_features = self.backbone.fc.in_features   # 2048 for ResNet50
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

        logger.info(
            f"ResNet50 initialized | pretrained={pretrained} | "
            f"freeze_backbone={freeze_backbone} | num_classes={num_classes}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_gradcam_layer(self):
        """Return the target layer for Grad-CAM (last conv block)."""
        return self.backbone.layer4


# ================================================================== #
# 3. VGG-16 (Transfer Learning)
# ================================================================== #
class VGG16Classifier(nn.Module):
    """
    VGG-16 fine-tuned for medical image classification.

    Grad-CAM target: model.backbone.features[-1]  (last MaxPool / Conv)
    Note: VGG is computationally heavier than ResNet.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.5,
    ):
        super().__init__()
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.vgg16(weights=weights)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen — training head only.")

        # Replace the classifier head
        in_features = self.backbone.classifier[6].in_features  # 4096
        self.backbone.classifier[6] = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

        logger.info(
            f"VGG16 initialized | pretrained={pretrained} | "
            f"freeze_backbone={freeze_backbone} | num_classes={num_classes}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_gradcam_layer(self):
        """Return last conv layer for Grad-CAM."""
        return self.backbone.features[-3]   # Conv2d before last MaxPool


# ================================================================== #
# 4. EfficientNet-B0
# ================================================================== #
class EfficientNetB0Classifier(nn.Module):
    """
    EfficientNet-B0 — lightweight and accurate.
    Best choice for resource-constrained environments.

    Grad-CAM target: model.backbone.features[-1]
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.4,
    ):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen — training head only.")

        in_features = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )

        logger.info(
            f"EfficientNetB0 initialized | pretrained={pretrained} | "
            f"freeze_backbone={freeze_backbone} | num_classes={num_classes}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_gradcam_layer(self):
        return self.backbone.features[-1]


# ================================================================== #
# 5. DenseNet-121
# ================================================================== #
class DenseNet121Classifier(nn.Module):
    """
    DenseNet-121 fine-tuned for medical image classification.

    DenseNet uses dense connections (each layer receives feature maps
    from all preceding layers), promoting feature reuse and reducing
    parameter count.

    Grad-CAM target: model.backbone.features.denseblock4
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.5,
    ):
        super().__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.densenet121(weights=weights)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen — training head only.")

        # Replace the classifier head
        in_features = self.backbone.classifier.in_features  # 1024
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

        logger.info(
            f"DenseNet121 initialized | pretrained={pretrained} | "
            f"freeze_backbone={freeze_backbone} | num_classes={num_classes}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_gradcam_layer(self):
        """Return the target layer for Grad-CAM (last dense block)."""
        return self.backbone.features.denseblock4


# ================================================================== #
# Model Factory
# ================================================================== #
MODEL_REGISTRY = {
    "custom_cnn":       CustomCNN,
    "resnet50":         ResNet50Classifier,
    "vgg16":            VGG16Classifier,
    "efficientnet_b0":  EfficientNetB0Classifier,
    "densenet121":      DenseNet121Classifier,
}


def build_model(cfg: dict) -> nn.Module:
    """
    Factory function: build model from config.

    Example config keys used:
        training.model        → model name
        training.pretrained   → bool
        training.freeze_backbone → bool
        dataset.num_classes   → int
    """
    model_name    = cfg["training"]["model"]
    num_classes   = cfg["dataset"]["num_classes"]
    pretrained    = cfg["training"].get("pretrained", True)
    freeze_backbone = cfg["training"].get("freeze_backbone", False)

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    ModelClass = MODEL_REGISTRY[model_name]

    # CustomCNN doesn't accept pretrained/freeze_backbone kwargs
    if model_name == "custom_cnn":
        model = ModelClass(num_classes=num_classes)
    else:
        model = ModelClass(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model: {model_name} | "
        f"Total params: {total_params:,} | "
        f"Trainable: {trainable_params:,}"
    )
    return model
