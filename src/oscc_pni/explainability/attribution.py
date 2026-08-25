"""Complementary local attribution methods for ROI-guided models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch import nn


@dataclass
class CamTarget:
    layer: nn.Module
    reshape_transform: Callable[[torch.Tensor], torch.Tensor] | None = None


class SingleInstanceMILWrapper(nn.Module):
    """Expose one-instance MIL outputs as a conventional image batch for CAM."""

    def __init__(self, mil_model: nn.Module) -> None:
        super().__init__()
        self.model = mil_model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        from oscc_pni.models.published_mil import forward_logits

        outputs = [forward_logits(self.model, image.unsqueeze(0)) for image in images]
        return torch.stack(outputs)


def _network(model: nn.Module) -> nn.Module:
    if hasattr(model, "encoder") and hasattr(model.encoder, "network"):
        return model.encoder.network
    if hasattr(model, "vit_backbone"):
        return model.vit_backbone
    if hasattr(model, "backbone"):
        return model.backbone
    if hasattr(model, "feature_extractor"):
        return model.feature_extractor
    return model


def _vit_reshape(tensor: torch.Tensor) -> torch.Tensor:
    # Remove class token and reshape 14 x 14 patch tokens for 224-pixel ViT-B/16.
    tokens = tensor[:, 1:, :]
    side = int(round(tokens.shape[1] ** 0.5))
    if side * side != tokens.shape[1]:
        raise ValueError(f"Cannot reshape {tokens.shape[1]} ViT patch tokens into a square.")
    return tokens.reshape(tokens.shape[0], side, side, tokens.shape[2]).permute(0, 3, 1, 2)


def _swin_reshape(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4:  # Some timm versions expose channels-last feature maps.
        return tensor.permute(0, 3, 1, 2)
    side = int(round(tensor.shape[1] ** 0.5))
    return tensor.reshape(tensor.shape[0], side, side, tensor.shape[2]).permute(0, 3, 1, 2)


def cam_target(model: nn.Module, backbone: str) -> CamTarget:
    network = _network(model)
    if backbone in {"resnet101", "resnet152_mil"}:
        return CamTarget(network.layer4[-1])
    if backbone in {"densenet121", "densenet121_mil"}:
        return CamTarget(network.features.denseblock4)
    if backbone in {"vit_base", "vit_base_mil"}:
        return CamTarget(network.blocks[-1].norm1, _vit_reshape)
    if backbone in {"swin_base", "swin_base_mil"}:
        stages = network.layers if hasattr(network, "layers") else network.stages
        return CamTarget(stages[-1].blocks[-1].norm1, _swin_reshape)
    raise ValueError(f"Unsupported backbone: {backbone}")


def smooth_gradcam_pp(
    model: nn.Module,
    image_tensor: torch.Tensor,
    *,
    backbone: str,
    target_class: int = 1,
    mil_single_instance: bool = False,
) -> np.ndarray:
    """Generate Smooth Grad-CAM++ via augmentation-smoothed GradCAM++."""
    try:
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError as exc:
        raise ImportError("Install the 'xai' extra for Grad-CAM-family methods.") from exc

    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    cam_model = SingleInstanceMILWrapper(model) if mil_single_instance else model
    target = cam_target(model, backbone)
    with GradCAMPlusPlus(
        model=cam_model,
        target_layers=[target.layer],
        reshape_transform=target.reshape_transform,
    ) as method:
        maps = method(
            input_tensor=image_tensor,
            targets=[ClassifierOutputTarget(target_class)] * len(image_tensor),
            aug_smooth=True,
        )
    return np.asarray(maps)


def integrated_gradients(
    model: nn.Module,
    image_tensor: torch.Tensor,
    *,
    target_class: int = 1,
    steps: int = 400,
    baseline: float = 0.0,
    mil_single_instance: bool = False,
) -> torch.Tensor:
    try:
        from captum.attr import IntegratedGradients
    except ImportError as exc:
        raise ImportError("Install the 'xai' extra for Captum attribution.") from exc
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    attribution_model = SingleInstanceMILWrapper(model) if mil_single_instance else model
    baselines = torch.full_like(image_tensor, baseline)
    return IntegratedGradients(attribution_model).attribute(
        image_tensor,
        baselines=baselines,
        target=target_class,
        n_steps=steps,
    )


def guided_gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    *,
    backbone: str,
    target_class: int = 1,
    mil_single_instance: bool = False,
) -> torch.Tensor:
    """Generate high-resolution Guided Grad-CAM attribution."""
    try:
        from captum.attr import GuidedGradCam
    except ImportError as exc:
        raise ImportError("Install the 'xai' extra for Captum attribution.") from exc
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    attribution_model = SingleInstanceMILWrapper(model) if mil_single_instance else model
    target = cam_target(model, backbone)
    return GuidedGradCam(attribution_model, target.layer).attribute(
        image_tensor,
        target=target_class,
        interpolate_mode="bilinear",
    )


def occlusion_sensitivity(
    model: nn.Module,
    image_tensor: torch.Tensor,
    *,
    target_class: int = 1,
    window: tuple[int, int, int] = (3, 15, 15),
    stride: tuple[int, int, int] = (3, 8, 8),
    baseline: float = 0.0,
    mil_single_instance: bool = False,
) -> torch.Tensor:
    try:
        from captum.attr import Occlusion
    except ImportError as exc:
        raise ImportError("Install the 'xai' extra for Captum attribution.") from exc
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    attribution_model = SingleInstanceMILWrapper(model) if mil_single_instance else model
    return Occlusion(attribution_model).attribute(
        image_tensor,
        target=target_class,
        sliding_window_shapes=window,
        strides=stride,
        baselines=baseline,
    )


def gradient_shap(
    model: nn.Module,
    image_tensor: torch.Tensor,
    background: torch.Tensor,
    *,
    target_class: int = 1,
    samples: int = 50,
    stdev: float = 0.0,
    mil_single_instance: bool = False,
) -> torch.Tensor:
    """Captum GradientShap with training images as the reference distribution."""
    try:
        from captum.attr import GradientShap
    except ImportError as exc:
        raise ImportError("Install the 'xai' extra for Captum attribution.") from exc
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    attribution_model = SingleInstanceMILWrapper(model) if mil_single_instance else model
    return GradientShap(attribution_model).attribute(
        image_tensor,
        baselines=background,
        target=target_class,
        n_samples=samples,
        stdevs=stdev,
    )


def normalize_attribution(
    attribution: torch.Tensor | np.ndarray, absolute: bool = True
) -> np.ndarray:
    values = (
        attribution.detach().cpu().numpy()
        if isinstance(attribution, torch.Tensor)
        else np.asarray(attribution)
    )
    while values.ndim > 2:
        values = np.mean(np.abs(values) if absolute else values, axis=0)
    minimum, maximum = np.nanmin(values), np.nanmax(values)
    if not np.isfinite(minimum) or maximum <= minimum:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - minimum) / (maximum - minimum)).astype(np.float32)


def overlay_heatmap(
    image: Image.Image | np.ndarray,
    heatmap: np.ndarray,
    *,
    alpha: float = 0.45,
) -> Image.Image:
    import matplotlib.cm as cm

    rgb = (
        np.asarray(
            image.convert("RGB") if isinstance(image, Image.Image) else image, dtype=np.float32
        )
        / 255.0
    )
    heat = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255)).resize(
        (rgb.shape[1], rgb.shape[0]), Image.Resampling.BILINEAR
    )
    colored = cm.get_cmap("jet")(np.asarray(heat, dtype=np.float32) / 255.0)[..., :3]
    overlay = np.clip((1.0 - alpha) * rgb + alpha * colored, 0.0, 1.0)
    return Image.fromarray(np.uint8(overlay * 255))
