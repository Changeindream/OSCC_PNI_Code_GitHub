"""Attention-based patient-level multiple-instance learning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from oscc_pni.config import normalize_backbone_name
from oscc_pni.models.backbones import BackboneEncoder, checkpoint_state


class AttentionPool(nn.Module):
    """Ilse-style normalized instance attention for a variable-length bag."""

    def __init__(self, feature_dim: int, attention_dim: int = 128) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError(
                "MIL features must have shape [instances, features] with at least one instance."
            )
        scores = self.scorer(features).squeeze(-1)
        weights = torch.softmax(scores, dim=0)
        pooled = torch.sum(features * weights.unsqueeze(-1), dim=0)
        return pooled, weights


class AttentionMIL(nn.Module):
    """Backbone encoder, normalized slice attention, and patient classifier."""

    def __init__(
        self,
        backbone: str,
        *,
        attention_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.10,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = normalize_backbone_name(backbone)
        self.encoder = BackboneEncoder(self.backbone_name, pretrained=pretrained)
        self.attention = AttentionPool(self.encoder.feature_dim, attention_dim)
        hidden_dim = max(attention_dim, self.encoder.feature_dim // 2)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        images: torch.Tensor,
        *,
        return_attention: bool = False,
        return_features: bool = False,
    ):
        """Predict one patient from `[instances, channels, height, width]`."""
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4:
            raise ValueError(f"Expected a 4-D patient bag, received {tuple(images.shape)}")
        features = self.encoder(images)
        patient_feature, weights = self.attention(features)
        logits = self.classifier(patient_feature)
        outputs: list[torch.Tensor] = [logits]
        if return_attention:
            outputs.append(weights)
        if return_features:
            outputs.append(features)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def load_slice_encoder(
        self, checkpoint: str | Path | dict[str, Any]
    ) -> tuple[list[str], list[str]]:
        """Initialize the MIL encoder from a checkpoint created by `SliceClassifier`."""
        state = checkpoint_state(checkpoint)
        encoder_state = {
            key.removeprefix("encoder."): value
            for key, value in state.items()
            if key.startswith("encoder.")
        }
        if not encoder_state:
            raise ValueError(
                "No 'encoder.' keys were found. Use a checkpoint produced by this repository "
                "or explicitly convert the original checkpoint."
            )
        incompatible = self.encoder.load_state_dict(encoder_state, strict=True)
        return list(incompatible.missing_keys), list(incompatible.unexpected_keys)


def create_mil_model(
    backbone: str,
    *,
    attention_dim: int = 128,
    num_classes: int = 2,
    dropout: float = 0.10,
    pretrained: bool = True,
    slice_checkpoint: str | Path | None = None,
) -> AttentionMIL:
    model = AttentionMIL(
        backbone,
        attention_dim=attention_dim,
        num_classes=num_classes,
        dropout=dropout,
        pretrained=pretrained,
    )
    if slice_checkpoint is not None:
        model.load_slice_encoder(slice_checkpoint)
    return model
