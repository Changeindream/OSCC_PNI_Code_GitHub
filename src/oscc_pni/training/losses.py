"""Loss functions and mixed-sample helpers."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """Binary/multiclass focal loss operating on unnormalized logits."""

    def __init__(self, gamma: float = 2.0, alpha: float | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.long()
        cross_entropy = F.cross_entropy(logits, targets, reduction="none")
        probability = torch.softmax(logits, dim=-1).gather(1, targets.view(-1, 1)).squeeze(1)
        loss = (1.0 - probability).pow(self.gamma) * cross_entropy
        if self.alpha is not None:
            alpha_factor = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
            loss = loss * alpha_factor
        return loss.mean()


class CrossEntropyPlusFocal(nn.Module):
    """ViT loss from Supplementary Table S3: CE plus 0.3 focal."""

    def __init__(self, focal_weight: float = 0.3, gamma: float = 2.0) -> None:
        super().__init__()
        self.focal_weight = focal_weight
        self.focal = FocalLoss(gamma=gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets) + self.focal_weight * self.focal(logits, targets)


def mixup(
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return images, labels, labels, 1.0
    lam = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(images.shape[0], device=images.device)
    return lam * images + (1.0 - lam) * images[permutation], labels, labels[permutation], lam


def cutmix(
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return images, labels, labels, 1.0
    lam = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(images.shape[0], device=images.device)
    height, width = images.shape[-2:]
    cut_ratio = np.sqrt(1.0 - lam)
    cut_width, cut_height = int(width * cut_ratio), int(height * cut_ratio)
    center_x, center_y = np.random.randint(width), np.random.randint(height)
    x1 = int(np.clip(center_x - cut_width // 2, 0, width))
    x2 = int(np.clip(center_x + cut_width // 2, 0, width))
    y1 = int(np.clip(center_y - cut_height // 2, 0, height))
    y2 = int(np.clip(center_y + cut_height // 2, 0, height))
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
    adjusted_lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(width * height))
    return mixed, labels, labels[permutation], adjusted_lam


def mixed_loss(
    criterion: nn.Module,
    logits: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    return lam * criterion(logits, first) + (1.0 - lam) * criterion(logits, second)
