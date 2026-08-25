from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from oscc_pni.data.manifest import stratified_patient_split
from oscc_pni.data.preprocessing import mask_bbox, window_and_scale
from oscc_pni.evaluation.metrics import binary_metrics, select_threshold
from oscc_pni.models.mil import AttentionPool
from oscc_pni.privacy import audit_manifest_frame


def test_window_and_scale_uses_reported_bounds() -> None:
    image = np.array([-200.0, -135.0, 40.0, 215.0, 400.0])
    result = window_and_scale(image)
    assert np.allclose(result[[0, 1]], 0.0)
    assert np.isclose(result[2], 0.5)
    assert np.allclose(result[[3, 4]], 1.0)


def test_mask_bbox_has_exclusive_upper_bounds() -> None:
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[3:6, 4:9] = 1
    assert mask_bbox(mask) == (3, 6, 4, 9)
    assert mask_bbox(mask, padding=2) == (1, 8, 2, 11)


def test_attention_weights_are_normalized() -> None:
    pool = AttentionPool(feature_dim=8, attention_dim=4)
    pooled, weights = pool(torch.randn(7, 8))
    assert pooled.shape == (8,)
    assert weights.shape == (7,)
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)


def test_manifest_audit_detects_patient_leakage() -> None:
    frame = pd.DataFrame(
        {
            "patient_id": ["CASE001", "CASE001"],
            "image_path": ["a.png", "b.png"],
            "label": [1, 1],
            "split": ["train", "test"],
        }
    )
    report = audit_manifest_frame(frame, check_files=False)
    assert not report.ok
    assert any("leakage" in error.lower() for error in report.errors)


def test_binary_metrics_and_validation_threshold() -> None:
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.3, 0.7, 0.9])
    threshold = select_threshold(y_true, probabilities)
    metrics = binary_metrics(y_true, probabilities, threshold=threshold)
    assert metrics["accuracy"] == 1.0
    assert metrics["tn"] == 2
    assert metrics["tp"] == 2


def test_stratified_split_never_splits_a_patient() -> None:
    rows = []
    for patient in range(20):
        for slice_index in range(2):
            rows.append(
                {
                    "patient_id": f"CASE{patient:03d}",
                    "image_path": f"CASE{patient:03d}_{slice_index}.png",
                    "label": patient % 2,
                    "center": "A" if patient < 10 else "B",
                }
            )
    result = stratified_patient_split(pd.DataFrame(rows), seed=7)
    assert result.groupby("patient_id")["split"].nunique().max() == 1
    assert set(result["split"]) == {"train", "validation", "test"}
