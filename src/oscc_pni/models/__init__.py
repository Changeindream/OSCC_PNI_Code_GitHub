"""Paper-aligned slice classifiers and attention MIL models."""

from .backbones import BackboneEncoder, SliceClassifier, create_slice_model
from .mil import AttentionMIL, create_mil_model
from .published_mil import (
    DenseNet121MIL,
    ResNet152MIL,
    SwinBaseMIL,
    ViTBaseMIL,
    create_published_mil_model,
)

__all__ = [
    "AttentionMIL",
    "BackboneEncoder",
    "DenseNet121MIL",
    "ResNet152MIL",
    "SliceClassifier",
    "SwinBaseMIL",
    "ViTBaseMIL",
    "create_mil_model",
    "create_published_mil_model",
    "create_slice_model",
]
