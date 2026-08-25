"""CECT resampling, soft-tissue windowing, and ROI-slice export."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def window_and_scale(
    image: np.ndarray,
    hu_min: float = -135.0,
    hu_max: float = 215.0,
) -> np.ndarray:
    """Clip a CT array to the reported soft-tissue window and scale to [0, 1]."""
    if hu_max <= hu_min:
        raise ValueError("hu_max must be greater than hu_min.")
    clipped = np.clip(np.asarray(image, dtype=np.float32), hu_min, hu_max)
    return (clipped - hu_min) / (hu_max - hu_min)


def mask_bbox(mask: np.ndarray, padding: int = 0) -> tuple[int, int, int, int]:
    """Return `(y_min, y_max, x_min, x_max)` with exclusive upper bounds."""
    coordinates = np.argwhere(mask > 0)
    if coordinates.size == 0:
        raise ValueError("Cannot compute an ROI bounding box from an empty mask.")
    y_min, x_min = coordinates.min(axis=0)
    y_max, x_max = coordinates.max(axis=0) + 1
    return (
        max(0, int(y_min) - padding),
        min(mask.shape[0], int(y_max) + padding),
        max(0, int(x_min) - padding),
        min(mask.shape[1], int(x_max) + padding),
    )


def paired_nifti_files(
    images_dir: str | Path, masks_dir: str | Path
) -> list[tuple[str, Path, Path]]:
    """Pair `.nii`/`.nii.gz` volumes by exact stem and reject ambiguous inputs."""
    image_map = {_nifti_stem(path): path for path in _nifti_paths(images_dir)}
    mask_map = {_nifti_stem(path): path for path in _nifti_paths(masks_dir)}
    missing_masks = sorted(set(image_map) - set(mask_map))
    missing_images = sorted(set(mask_map) - set(image_map))
    if missing_masks or missing_images:
        raise ValueError(
            f"Unpaired volumes. Missing masks={missing_masks[:5]}, missing images={missing_images[:5]}"
        )
    return [(stem, image_map[stem], mask_map[stem]) for stem in sorted(image_map)]


def _nifti_paths(directory: str | Path) -> list[Path]:
    root = Path(directory)
    paths = sorted(root.glob("*.nii")) + sorted(root.glob("*.nii.gz"))
    if not paths:
        raise FileNotFoundError(f"No NIfTI files found in {root}")
    return paths


def _nifti_stem(path: Path) -> str:
    return path.name[:-7] if path.name.lower().endswith(".nii.gz") else path.stem


def _resample_pair(image_path: Path, mask_path: Path, spacing: tuple[float, float, float]):
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise ImportError("Install the 'medical-images' extra to prepare NIfTI data.") from exc

    image = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
    mask = sitk.ReadImage(str(mask_path))
    new_size = [
        max(1, int(round(size * old_spacing / new_spacing)))
        for size, old_spacing, new_spacing in zip(
            image.GetSize(), image.GetSpacing(), spacing, strict=False
        )
    ]

    def resample(source, interpolator):
        return sitk.Resample(
            source,
            new_size,
            sitk.Transform(),
            interpolator,
            image.GetOrigin(),
            spacing,
            image.GetDirection(),
            0,
            source.GetPixelID(),
        )

    return resample(image, sitk.sitkLinear), resample(mask, sitk.sitkNearestNeighbor)


def export_tumor_slices(
    *,
    images_dir: str | Path,
    masks_dir: str | Path,
    output_dir: str | Path,
    id_map: Mapping[str, str],
    label: int,
    split: str,
    center: str | None = None,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    hu_min: float = -135.0,
    hu_max: float = 215.0,
    padding: int = 8,
    apply_mask: bool = True,
    image_size: int = 224,
) -> pd.DataFrame:
    """Export all tumor-bearing axial slices and return manifest rows.

    `id_map` must map each private source stem to a de-identified identifier. Source
    stems are never written to output filenames or the returned manifest.
    """
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise ImportError("Install the 'medical-images' extra to prepare NIfTI data.") from exc

    if label not in (0, 1):
        raise ValueError("label must be 0 or 1.")
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test.")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for private_stem, image_path, mask_path in paired_nifti_files(images_dir, masks_dir):
        if private_stem not in id_map:
            raise KeyError(f"No de-identified patient_id mapping for source stem: {private_stem}")
        patient_id = str(id_map[private_stem]).strip()
        image_itk, mask_itk = _resample_pair(image_path, mask_path, spacing)
        image = sitk.GetArrayFromImage(image_itk)  # z, y, x
        mask = sitk.GetArrayFromImage(mask_itk) > 0
        scaled = window_and_scale(image, hu_min, hu_max)

        for slice_index in np.flatnonzero(mask.reshape(mask.shape[0], -1).any(axis=1)):
            slice_mask = mask[slice_index]
            y_min, y_max, x_min, x_max = mask_bbox(slice_mask, padding)
            crop = scaled[slice_index, y_min:y_max, x_min:x_max]
            crop_mask = slice_mask[y_min:y_max, x_min:x_max]
            if apply_mask:
                crop = crop * crop_mask
            output_image = Image.fromarray(np.round(crop * 255).astype(np.uint8), mode="L")
            output_image = output_image.resize((image_size, image_size), Image.Resampling.BILINEAR)
            filename = f"{patient_id}_slice_{int(slice_index):04d}.png"
            output_path = destination / filename
            output_image.save(output_path, optimize=True)
            row: dict[str, object] = {
                "patient_id": patient_id,
                "image_path": str(output_path.resolve()),
                "label": label,
                "split": split,
                "slice_index": int(slice_index),
                "roi_area": int(slice_mask.sum()),
            }
            if center:
                row["center"] = center
            rows.append(row)

    return pd.DataFrame(rows)


def load_id_map(path: str | Path) -> dict[str, str]:
    frame = pd.read_csv(path, dtype=str)
    required = {"source_stem", "patient_id"}
    if not required.issubset(frame.columns):
        raise ValueError("ID map must contain source_stem and patient_id columns.")
    if frame[list(required)].isna().any().any():
        raise ValueError("ID map contains missing source_stem or patient_id values.")
    if frame["source_stem"].duplicated().any() or frame["patient_id"].duplicated().any():
        raise ValueError("ID map must be one-to-one.")
    return dict(zip(frame["source_stem"], frame["patient_id"], strict=False))
