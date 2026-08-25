"""PyRadiomics extraction with the parameters reported in Supplementary Methods S3."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from oscc_pni.data.preprocessing import paired_nifti_files


def build_radiomics_extractor(
    *,
    bin_width_hu: float = 25.0,
    log_sigma_mm: tuple[float, ...] = (1.0, 3.0, 5.0),
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Any:
    """Create a PyRadiomics 3.0 extractor for original, LoG, and wavelet images."""
    try:
        from radiomics import featureextractor
    except ImportError as exc:
        raise ImportError("Install the 'radiomics' extra to extract handcrafted features.") from exc

    extractor = featureextractor.RadiomicsFeatureExtractor(
        binWidth=bin_width_hu,
        sigma=list(log_sigma_mm),
        resampledPixelSpacing=list(spacing_mm),
        interpolator="sitkBSpline",
        normalize=False,
    )
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName("Original")
    extractor.enableImageTypeByName("LoG", customArgs={"sigma": list(log_sigma_mm)})
    extractor.enableImageTypeByName("Wavelet")
    extractor.enableAllFeatures()
    return extractor


def extract_radiomics_directory(
    *,
    images_dir: str | Path,
    masks_dir: str | Path,
    id_map: Mapping[str, str],
    label_map: Mapping[str, int] | None = None,
    extractor: Any | None = None,
) -> pd.DataFrame:
    """Extract one row per paired 3-D image/mask without retaining private filenames."""
    extractor = extractor or build_radiomics_extractor()
    rows: list[dict[str, object]] = []
    for private_stem, image_path, mask_path in paired_nifti_files(images_dir, masks_dir):
        if private_stem not in id_map:
            raise KeyError(f"No de-identified patient_id mapping for {private_stem}")
        values = extractor.execute(str(image_path), str(mask_path))
        row: dict[str, object] = {"patient_id": str(id_map[private_stem])}
        if label_map is not None:
            if private_stem not in label_map:
                raise KeyError(f"No label mapping for {private_stem}")
            row["label"] = int(label_map[private_stem])
        for name, value in values.items():
            if name.startswith("diagnostics_"):
                continue
            if hasattr(value, "item"):
                value = value.item()
            row[name] = value
        rows.append(row)
    return pd.DataFrame(rows)
