"""Frozen Kinetics-400 ResNet-18 feature extraction for masked three-dimensional CT."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.models.video import R3D_18_Weights, r3d_18

from oscc_pni.data.preprocessing import paired_nifti_files, window_and_scale


class FrozenR3D18Extractor(nn.Module):
    """A single-channel, 512-dimensional frozen R3D-18 encoder."""

    feature_dim = 512

    def __init__(self, *, pretrained: bool = True) -> None:
        super().__init__()
        weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
        network = r3d_18(weights=weights)
        original = network.stem[0]
        single_channel = nn.Conv3d(
            1,
            original.out_channels,
            kernel_size=original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            bias=False,
        )
        if pretrained:
            with torch.no_grad():
                single_channel.weight.copy_(original.weight.mean(dim=1, keepdim=True))
        network.stem[0] = single_channel
        network.fc = nn.Identity()
        self.network = network.eval()
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """Encode `[N, 1, depth, height, width]` volumes."""
        return self.network(volume)


def _load_masked_volume(
    image_path: Path,
    mask_path: Path,
    *,
    hu_min: float,
    hu_max: float,
) -> torch.Tensor:
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise ImportError("Install the 'medical-images' extra to read NIfTI volumes.") from exc

    image = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path), sitk.sitkFloat32))
    mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))) > 0
    if image.shape != mask.shape:
        raise ValueError(f"Image/mask shape mismatch: {image.shape} vs {mask.shape}")
    if not np.any(mask):
        raise ValueError(f"Empty mask: {mask_path}")
    # Preserve the original extractor's full-volume geometry and zero the
    # extratumoral voxels. Upstream NIfTI files should already be resampled to
    # 1-mm isotropic spacing, as documented in the paper configuration.
    masked = window_and_scale(image, hu_min, hu_max) * mask
    return torch.from_numpy(masked.astype(np.float32))[None, None]


@torch.inference_mode()
def extract_deep_features_directory(
    *,
    images_dir: str | Path,
    masks_dir: str | Path,
    id_map: Mapping[str, str],
    label_map: Mapping[str, int] | None = None,
    model: FrozenR3D18Extractor | None = None,
    device: str | torch.device | None = None,
    hu_min: float = -135.0,
    hu_max: float = 215.0,
) -> pd.DataFrame:
    """Extract one frozen 512-D feature vector per paired volume."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = (model or FrozenR3D18Extractor()).to(device).eval()
    rows: list[dict[str, object]] = []

    for private_stem, image_path, mask_path in paired_nifti_files(images_dir, masks_dir):
        if private_stem not in id_map:
            raise KeyError(f"No de-identified patient_id mapping for {private_stem}")
        volume = _load_masked_volume(image_path, mask_path, hu_min=hu_min, hu_max=hu_max).to(device)
        vector = model(volume).squeeze(0).detach().cpu().numpy()
        row: dict[str, object] = {"patient_id": str(id_map[private_stem])}
        if label_map is not None:
            row["label"] = int(label_map[private_stem])
        row.update({f"feature_{index}": float(value) for index, value in enumerate(vector)})
        rows.append(row)
    return pd.DataFrame(rows)
