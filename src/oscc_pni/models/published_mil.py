"""Checkpoint-compatible MIL models used by the published Gradio interface.

These definitions intentionally preserve the parameter names and tensor shapes of
the four original training implementations.  The shared training API in
``models.mil`` remains available for new experiments; this module is the source
of truth for loading the released inference checkpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision import models


class _AttentionPoolingMIL(nn.Module):
    """Common ResNet/DenseNet attention pooling with legacy-compatible keys."""

    feature_extractor: nn.Module

    def _build_head(self, feature_dim: int, attention_dim: int, num_classes: int) -> None:
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward_with_details(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4 or images.shape[0] == 0:
            raise ValueError("A patient bag must have shape [slices, channels, height, width].")
        features = self.feature_extractor(images)
        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)
        attention = torch.softmax(self.attention_net(features).squeeze(-1), dim=0)
        pooled = torch.sum(features * attention.unsqueeze(-1), dim=0)
        return self.classifier(pooled), attention, features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.forward_with_details(images)
        return logits


class ResNet152MIL(_AttentionPoolingMIL):
    """ResNet-152 patient-level MIL model from the deployed interface."""

    def __init__(self, *, attention_dim: int = 128, num_classes: int = 2) -> None:
        super().__init__()
        backbone = models.resnet152(weights=None)
        feature_dim = int(backbone.fc.in_features)
        backbone.fc = nn.Identity()
        self.feature_extractor = backbone
        self._build_head(feature_dim, attention_dim, num_classes)


class DenseNet121MIL(_AttentionPoolingMIL):
    """DenseNet-121 patient-level MIL model from the deployed interface."""

    def __init__(self, *, attention_dim: int = 128, num_classes: int = 2) -> None:
        super().__init__()
        backbone = models.densenet121(weights=None)
        feature_dim = int(backbone.classifier.in_features)
        backbone.classifier = nn.Identity()
        self.feature_extractor = backbone
        self._build_head(feature_dim, attention_dim, num_classes)


class SwinMILAttention(nn.Module):
    """Multi-head contextualization followed by normalized instance attention."""

    def __init__(self, feature_dim: int, attention_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        if attention_dim % num_heads:
            raise ValueError("attention_dim must be divisible by num_heads")
        self.feature_dim = feature_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.query_projection = nn.Linear(feature_dim, attention_dim)
        self.key_projection = nn.Linear(feature_dim, attention_dim)
        self.value_projection = nn.Linear(feature_dim, attention_dim)
        self.output_projection = nn.Linear(attention_dim, feature_dim)
        self.attention_weights = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )
        self.gate = nn.Sequential(nn.Linear(feature_dim, attention_dim), nn.Sigmoid())
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.scale = self.head_dim**-0.5

    def forward(
        self, features: torch.Tensor, *, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size, instances, _ = features.shape
        shape = (batch_size, instances, self.num_heads, self.head_dim)
        query = self.query_projection(features).view(shape).transpose(1, 2)
        key = self.key_projection(features).view(shape).transpose(1, 2)
        value = self.value_projection(features).view(shape).transpose(1, 2)
        probabilities = torch.softmax((query @ key.transpose(-2, -1)) * self.scale, dim=-1)
        context = (self.dropout(probabilities) @ value).transpose(1, 2).contiguous()
        context = context.view(batch_size, instances, self.attention_dim)
        contextualized = self.layer_norm(self.output_projection(context) + features)
        attention = torch.softmax(self.attention_weights(contextualized), dim=1)
        gated = contextualized * self.gate(contextualized)
        pooled = torch.sum(gated * attention, dim=1)
        if return_attention:
            return pooled, attention.squeeze(-1)
        return pooled


class SwinBaseMIL(nn.Module):
    """Legacy Swin-Base MIL architecture required by the released checkpoint."""

    def __init__(
        self,
        *,
        attention_dim: int = 256,
        num_classes: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        from oscc_pni.models.legacy_swin import swin_base_patch4_window7_224

        self.num_classes = num_classes
        self.use_gated = True
        self.backbone = swin_base_patch4_window7_224(pretrained=False, num_classes=0)
        self.feature_dim = 1024
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.feature_dim, attention_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(attention_dim, attention_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mil_attention = SwinMILAttention(attention_dim, attention_dim, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(attention_dim, attention_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(attention_dim // 2, num_classes),
        )

    def forward(
        self,
        bag: torch.Tensor,
        *,
        return_attention: bool = False,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if bag.ndim == 3:
            bag = bag.unsqueeze(0)
        if bag.ndim != 4 or bag.shape[0] == 0:
            raise ValueError("A patient bag must have shape [slices, channels, height, width].")
        # Process one slice at a time to match the original deployed implementation.
        instance_features = torch.cat([self.backbone(item.unsqueeze(0)) for item in bag], dim=0)
        transformed = self.feature_extractor(instance_features).unsqueeze(0)
        if return_attention:
            pooled, attention = self.mil_attention(transformed, return_attention=True)
        else:
            pooled = self.mil_attention(transformed)
            attention = None
        logits = self.classifier(pooled.squeeze(0))
        result: list[torch.Tensor] = [logits]
        if attention is not None:
            result.append(attention.squeeze(0))
        if return_features:
            result.append(transformed.squeeze(0))
        return result[0] if len(result) == 1 else tuple(result)

    def forward_with_details(
        self, bag: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, attention, features = self(bag, return_attention=True, return_features=True)
        return logits, attention, features


class ViTBaseMIL(nn.Module):
    """ViT-B/16 patient-level MIL architecture used by the released checkpoint."""

    def __init__(
        self,
        *,
        attention_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("Install timm to use the ViT-MIL checkpoint.") from exc
        self.model_name = "vit_base_patch16_224"
        self.num_classes = num_classes
        self.attention_dim = attention_dim
        self.vit_backbone = timm.create_model(
            self.model_name,
            pretrained=False,
            num_classes=0,
            drop_rate=dropout,
            attn_drop_rate=dropout,
            drop_path_rate=dropout,
        )
        self.feature_dim = int(self.vit_backbone.embed_dim)
        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim, attention_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attention_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim // 2, num_classes),
        )

    def forward(self, bag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if bag.ndim == 3:
            bag = bag.unsqueeze(0)
        if bag.ndim != 4 or bag.shape[0] == 0:
            raise ValueError("A patient bag must have shape [slices, channels, height, width].")
        # The slice-wise loop is retained because it is part of the deployed model path.
        features = torch.cat([self.vit_backbone(item.unsqueeze(0)) for item in bag], dim=0)
        attention = torch.softmax(self.attention(features), dim=0)
        pooled = torch.sum(features * attention, dim=0)
        return self.classifier(pooled), attention.squeeze(-1), features

    def forward_with_details(
        self, bag: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self(bag)


PUBLISHED_ARCHITECTURES = {
    "resnet152_mil": ResNet152MIL,
    "densenet121_mil": DenseNet121MIL,
    "swin_base_mil": SwinBaseMIL,
    "vit_base_mil": ViTBaseMIL,
}


def create_published_mil_model(architecture: str) -> nn.Module:
    """Create the checkpoint-compatible model registered under ``architecture``."""
    try:
        constructor = PUBLISHED_ARCHITECTURES[architecture]
    except KeyError as exc:
        choices = ", ".join(PUBLISHED_ARCHITECTURES)
        raise ValueError(
            f"Unknown published architecture '{architecture}'. Choose: {choices}"
        ) from exc
    return constructor()


def load_published_checkpoint(
    model: nn.Module, checkpoint: str | Path | dict[str, Any]
) -> dict[str, Any]:
    """Strictly load an inference checkpoint and return its non-tensor metadata."""
    payload = (
        torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        if isinstance(checkpoint, str | Path)
        else checkpoint
    )
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint must be a dictionary or path to a dictionary checkpoint.")
    state = payload.get("model_state_dict", payload.get("state_dict", payload))
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a model state dictionary.")
    model.load_state_dict(state, strict=True)
    return {
        key: value
        for key, value in payload.items()
        if key not in {"model_state_dict", "state_dict"}
    }


def forward_with_details(
    model: nn.Module, bag: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return logits, normalized slice attention, and instance features."""
    method = getattr(model, "forward_with_details", None)
    if method is None:
        raise TypeError(f"{type(model).__name__} does not expose MIL inference details.")
    logits, attention, features = method(bag)
    return logits, attention.reshape(-1), features


def forward_logits(model: nn.Module, bag: torch.Tensor) -> torch.Tensor:
    """Return only logits for CAM wrappers and other classifier-style consumers."""
    logits, _, _ = forward_with_details(model, bag)
    return logits
