"""Activation maximization and UMAP representation analysis."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from torch import nn


def total_variation(image: torch.Tensor) -> torch.Tensor:
    vertical = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
    horizontal = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]))
    return vertical + horizontal


def activation_maximization(
    model: nn.Module,
    target_layer: nn.Module,
    *,
    channel: int = 0,
    iterations: int = 30,
    learning_rate: float = 0.05,
    image_size: int = 224,
    tv_weight: float = 1e-4,
    l2_weight: float = 1e-5,
    device: torch.device | str | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Optimize an input that activates one selected convolutional channel."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    generator = torch.Generator(device=device).manual_seed(seed)
    image = torch.randn((1, 3, image_size, image_size), generator=generator, device=device) * 0.05
    image.requires_grad_(True)
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        captured["activation"] = output

    handle = target_layer.register_forward_hook(hook)
    optimizer = torch.optim.Adam([image], lr=learning_rate)
    model = model.to(device).eval()
    try:
        for _ in range(iterations):
            optimizer.zero_grad(set_to_none=True)
            model(image)
            activation = captured["activation"]
            if activation.ndim < 2 or channel >= activation.shape[1]:
                raise ValueError(
                    f"Channel {channel} is invalid for activation shape {tuple(activation.shape)}"
                )
            objective = activation[:, channel].mean()
            loss = -objective + tv_weight * total_variation(image) + l2_weight * image.pow(2).mean()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                image.clamp_(-3.0, 3.0)
    finally:
        handle.remove()

    result = image.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    result = np.power(
        np.clip((result - result.min()) / max(result.max() - result.min(), 1e-8), 0, 1), 1 / 2.2
    )
    low, high = np.percentile(result, [1, 99])
    return np.clip((result - low) / max(high - low, 1e-8), 0, 1)


@torch.inference_mode()
def collect_layer_features(
    model: nn.Module,
    target_layer: nn.Module,
    batches: Iterable[dict[str, torch.Tensor]],
    *,
    device: torch.device | str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect global-average-pooled target-layer features and labels."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        captured["activation"] = output

    handle = target_layer.register_forward_hook(hook)
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    model = model.to(device).eval()
    try:
        for batch in batches:
            images = batch["image"].to(device)
            model(images)
            activation = captured["activation"]
            if activation.ndim > 2:
                pooled = F.adaptive_avg_pool2d(activation, 1).flatten(1)
            else:
                pooled = activation
            features.append(pooled.cpu().numpy())
            labels.append(batch["label"].cpu().numpy())
    finally:
        handle.remove()
    return np.concatenate(features), np.concatenate(labels)


def umap_projection(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    seed: int = 42,
) -> tuple[pd.DataFrame, float]:
    """Standardize pooled features and return the reported two-dimensional UMAP."""
    try:
        import umap
    except ImportError as exc:
        raise ImportError("Install the 'xai' extra for UMAP analysis.") from exc
    standardized = StandardScaler().fit_transform(np.asarray(features))
    embedding = umap.UMAP(
        n_components=2,
        n_neighbors=neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    ).fit_transform(standardized)
    score = (
        float(silhouette_score(embedding, labels)) if np.unique(labels).size > 1 else float("nan")
    )
    return pd.DataFrame(
        {"umap_1": embedding[:, 0], "umap_2": embedding[:, 1], "label": labels}
    ), score
