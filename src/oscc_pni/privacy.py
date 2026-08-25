"""Privacy checks for manifests and output paths."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

SUSPICIOUS_PATTERNS = {
    "parenthesized_record_number": re.compile(r"\(\s*\d{5,}\s*\)"),
    "long_numeric_identifier": re.compile(r"(?<!\d)\d{7,}(?!\d)"),
    "probable_romanized_name": re.compile(
        r"(?:^|[/\\_])(?:[A-Z]{2,}[ _-]){1,4}[A-Z]{2,}(?:[/\\_(]|$)"
    ),
}


@dataclass
class AuditReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def format(self) -> str:
        lines = ["Manifest audit: " + ("PASS" if self.ok else "FAIL")]
        for key, value in self.summary.items():
            lines.append(f"  {key}: {value}")
        if self.errors:
            lines.append("Errors:")
            lines.extend(f"  - {item}" for item in self.errors)
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  - {item}" for item in self.warnings)
        return "\n".join(lines)


def suspicious_identifier(value: object) -> list[str]:
    text = str(value)
    return [name for name, pattern in SUSPICIOUS_PATTERNS.items() if pattern.search(text)]


def audit_manifest_frame(frame: pd.DataFrame, *, check_files: bool = True) -> AuditReport:
    report = AuditReport()
    required = {"patient_id", "image_path", "label", "split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        report.errors.append(f"Missing required columns: {', '.join(missing)}")
        return report

    normalized = frame.copy()
    normalized["patient_id"] = normalized["patient_id"].astype(str).str.strip()
    normalized["split"] = normalized["split"].astype(str).str.strip().str.lower()

    allowed_splits = {"train", "validation", "test"}
    unknown_splits = sorted(set(normalized["split"]) - allowed_splits)
    if unknown_splits:
        report.errors.append(f"Unknown split values: {unknown_splits}")

    labels = pd.to_numeric(normalized["label"], errors="coerce")
    if labels.isna().any() or not set(labels.dropna().astype(int)).issubset({0, 1}):
        report.errors.append("Labels must contain only binary values 0 and 1.")

    patient_split_counts = normalized.groupby("patient_id")["split"].nunique()
    overlaps = patient_split_counts[patient_split_counts > 1].index.tolist()
    if overlaps:
        report.errors.append(
            f"Patient leakage across splits: {len(overlaps)} patient(s), examples={overlaps[:5]}"
        )

    label_counts = normalized.groupby("patient_id")["label"].nunique()
    conflicts = label_counts[label_counts > 1].index.tolist()
    if conflicts:
        report.errors.append(
            f"Conflicting labels within patient: {len(conflicts)} patient(s), examples={conflicts[:5]}"
        )

    duplicated = normalized.duplicated(subset=["patient_id", "image_path"], keep=False)
    if duplicated.any():
        report.errors.append(f"Duplicate patient/slice rows: {int(duplicated.sum())}")

    suspicious: list[tuple[str, str]] = []
    for column in ("patient_id", "image_path", "mask_path"):
        if column not in normalized:
            continue
        for value in normalized[column].dropna().astype(str).unique():
            matches = suspicious_identifier(value)
            if matches:
                suspicious.append((value, ",".join(matches)))
    if suspicious:
        examples = [f"{value} ({reason})" for value, reason in suspicious[:5]]
        report.errors.append(
            "Potential identifiers detected in IDs or paths; de-identify before Git use. "
            f"Examples: {examples}"
        )

    if check_files:
        missing_files = [
            path for path in normalized["image_path"].astype(str) if not Path(path).is_file()
        ]
        if missing_files:
            report.errors.append(
                f"Missing image files: {len(missing_files)}, examples={missing_files[:3]}"
            )

    patient_summary = (
        normalized.drop_duplicates("patient_id").groupby("split", observed=False).size().to_dict()
    )
    slice_summary = normalized.groupby("split", observed=False).size().to_dict()
    report.summary = {
        "rows": len(normalized),
        "patients": normalized["patient_id"].nunique(),
        "patient_counts": patient_summary,
        "slice_counts": slice_summary,
    }
    return report


def audit_manifest(path: str | Path, *, check_files: bool = True) -> AuditReport:
    frame = pd.read_csv(path)
    return audit_manifest_frame(frame, check_files=check_files)
