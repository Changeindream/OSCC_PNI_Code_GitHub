"""Training entry points for single-slice and patient-level models."""

from .engine import evaluate_mil_checkpoint, train_mil_model, train_slice_model
from .losses import FocalLoss

__all__ = ["FocalLoss", "evaluate_mil_checkpoint", "train_mil_model", "train_slice_model"]
