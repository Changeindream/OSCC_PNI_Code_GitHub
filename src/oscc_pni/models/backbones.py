"""Unified feature encoders for the four reported two-dimensional backbones."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision import models

from oscc_pni.config import normalize_backbone_name

BACKBONE_DROPOUT = {
    "resnet101": 0.50,
    "densenet121": 0.10,
    "vit_base": 0.15,
    "swin_base": 0.10,
}


class BackboneEncoder(nn.Module):
    """Expose one feature-vector interface across CNN and transformer backbones."""

    def __init__(self, name: str, *, pretrained: bool = True) -> None:
        super().__init__()
        self.name = normalize_backbone_name(name)

        if self.name == "resnet101":
            weights = models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
            network = models.resnet101(weights=weights)
            self.feature_dim = network.fc.in_features
            network.fc = nn.Identity()
        elif self.name == "densenet121":
            weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            network = models.densenet121(weights=weights)
            self.feature_dim = network.classifier.in_features
            network.classifier = nn.Identity()
        else:
            try:
                import timm
            except ImportError as exc:
                raise ImportError("Install timm to use ViT or Swin Transformer.") from exc
            timm_name = {
                "vit_base": "vit_base_patch16_224",
                "swin_base": "swin_base_patch4_window7_224",
            }[self.name]
            network = timm.create_model(timm_name, pretrained=pretrained, num_classes=0)
            self.feature_dim = int(network.num_features)

        self.network = network

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.network(images)
        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)
        if features.ndim != 2:
            raise RuntimeError(f"Expected [N, D] features, received {tuple(features.shape)}")
        return features


class SliceClassifier(nn.Module):
    """ROI-guided single-slice classifier used before patient-level MIL."""

    def __init__(
        self,
        backbone: str,
        *,
        num_classes: int = 2,
        dropout: float | None = None,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        canonical = normalize_backbone_name(backbone)
        self.backbone_name = canonical
        self.encoder = BackboneEncoder(canonical, pretrained=pretrained)
        rate = BACKBONE_DROPOUT[canonical] if dropout is None else dropout
        self.classifier = nn.Sequential(
            nn.Dropout(rate),
            nn.Linear(self.encoder.feature_dim, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(images))


def create_slice_model(
    backbone: str,
    *,
    num_classes: int = 2,
    dropout: float | None = None,
    pretrained: bool = True,
) -> SliceClassifier:
    return SliceClassifier(
        backbone,
        num_classes=num_classes,
        dropout=dropout,
        pretrained=pretrained,
    )


def checkpoint_state(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Extract a state dictionary from common checkpoint containers."""
    if isinstance(checkpoint, str | Path):
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a path or dictionary.")
    for key in ("model_state_dict", "state_dict", "model"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            checkpoint = checkpoint[key]
            break
    return {
        (key.removeprefix("module.")): value
        for key, value in checkpoint.items()
        if isinstance(value, torch.Tensor)
    }


def load_checkpoint(
    model: nn.Module,
    checkpoint: str | Path | dict[str, Any],
    *,
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    state = checkpoint_state(checkpoint)
    incompatible = model.load_state_dict(state, strict=strict)
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)
