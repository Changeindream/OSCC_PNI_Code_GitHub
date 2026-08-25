"""Manifest loading, validation, and split summaries."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from oscc_pni.privacy import AuditReport, audit_manifest_frame

REQUIRED_COLUMNS = ("patient_id", "image_path", "label", "split")


def load_manifest(path: str | Path, *, resolve_paths: bool = True) -> pd.DataFrame:
    """Load a manifest and normalize its identifiers, labels, and split names."""
    manifest_path = Path(path).expanduser().resolve()
    frame = pd.read_csv(manifest_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Manifest is missing columns: {', '.join(missing)}")

    frame = frame.copy()
    frame["patient_id"] = frame["patient_id"].astype(str).str.strip()
    frame["split"] = frame["split"].astype(str).str.strip().str.lower()
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)

    if resolve_paths:
        for column in ("image_path", "mask_path"):
            if column not in frame:
                continue
            frame[column] = frame[column].map(
                lambda value: str(_resolve_data_path(value, manifest_path.parent))
                if pd.notna(value) and str(value).strip()
                else value
            )

    if "slice_index" in frame:
        frame["slice_index"] = pd.to_numeric(frame["slice_index"], errors="coerce")
    return frame


def _resolve_data_path(value: object, base_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def validate_manifest(
    frame: pd.DataFrame,
    *,
    check_files: bool = True,
    require_splits: Iterable[str] = ("train", "validation", "test"),
) -> AuditReport:
    report = audit_manifest_frame(frame, check_files=check_files)
    observed = set(frame["split"].astype(str).str.lower()) if "split" in frame else set()
    missing_splits = sorted(set(require_splits) - observed)
    if missing_splits:
        report.errors.append(f"Required splits are absent: {missing_splits}")
    return report


def manifest_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return patient, slice, and class counts for each split."""
    patient_rows = frame.drop_duplicates("patient_id")
    patient_counts = (
        patient_rows.groupby(["split", "label"], observed=False).size().rename("patients")
    )
    slice_counts = frame.groupby(["split", "label"], observed=False).size().rename("slices")
    summary = pd.concat([patient_counts, slice_counts], axis=1).fillna(0).astype(int)
    return summary.reset_index().sort_values(["split", "label"]).reset_index(drop=True)


def save_manifest(frame: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination


def stratified_patient_split(
    frame: pd.DataFrame,
    *,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign patients to 7:2:1-like splits within center-by-label strata.

    The allocation is performed once per patient, then joined back to every slice.
    Small strata may not contribute to every split; the returned summary should be
    inspected before training.
    """
    import numpy as np

    required = {"patient_id", "label", "center"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Stratified splitting requires columns: {sorted(required)}")
    ratios = np.asarray([train_ratio, validation_ratio, test_ratio], dtype=float)
    if np.any(ratios < 0) or not np.isclose(ratios.sum(), 1.0):
        raise ValueError("train, validation, and test ratios must be non-negative and sum to 1.")

    patient_rows = frame[["patient_id", "label", "center"]].drop_duplicates().copy()
    if patient_rows["patient_id"].duplicated().any():
        raise ValueError("Each patient must have exactly one label and center before splitting.")
    generator = np.random.default_rng(seed)
    assignments: list[dict[str, str]] = []
    names = np.asarray(["train", "validation", "test"])

    for (_center, _label), group in patient_rows.groupby(["center", "label"], sort=True):
        patient_ids = group["patient_id"].astype(str).to_numpy()
        generator.shuffle(patient_ids)
        positive = np.flatnonzero(ratios > 0)
        counts = np.zeros(3, dtype=int)
        if len(patient_ids) >= len(positive):
            # Preserve representation in every requested split when the stratum is large enough.
            counts[positive] = 1
            remaining = len(patient_ids) - len(positive)
            expected = remaining * ratios / ratios.sum()
            additions = np.floor(expected).astype(int)
            leftover = remaining - int(additions.sum())
            fractional_order = np.argsort(-(expected - additions), kind="stable")
            additions[fractional_order[:leftover]] += 1
            counts += additions
        else:
            # For very small strata, prioritize the largest requested proportions.
            counts[np.argsort(-ratios, kind="stable")[: len(patient_ids)]] = 1
        start = 0
        for split_name, count in zip(names, counts, strict=False):
            for patient_id in patient_ids[start : start + count]:
                assignments.append({"patient_id": patient_id, "split": str(split_name)})
            start += int(count)

    assignment_frame = pd.DataFrame(assignments)
    output = frame.drop(columns=["split"], errors="ignore").merge(
        assignment_frame, on="patient_id", how="left", validate="many_to_one"
    )
    if output["split"].isna().any():
        raise RuntimeError("Some patients were not assigned to a split.")
    return output
