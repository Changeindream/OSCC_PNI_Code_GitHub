"""Binary classification metrics with patient-level bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

THRESHOLD_METRICS = (
    "accuracy",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "f1",
    "mcc",
    "youden",
)


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def select_threshold(y_true: Sequence[int], probabilities: Sequence[float]) -> float:
    """Select the validation threshold that maximizes Youden's index."""
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if np.unique(y).size != 2:
        raise ValueError("Threshold selection requires both classes.")
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y, scores)
    youden = true_positive_rate - false_positive_rate
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    valid_indices = np.flatnonzero(finite)
    best = valid_indices[int(np.argmax(youden[finite]))]
    return float(np.clip(thresholds[best], 0.0, 1.0))


def binary_metrics(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    *,
    threshold: float,
) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if y.shape != scores.shape:
        raise ValueError("y_true and probabilities must have the same shape.")
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    result: dict[str, float | int] = {
        "n": int(len(y)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy_score(y, predictions)),
        "sensitivity": float(recall_score(y, predictions, pos_label=1, zero_division=0)),
        "specificity": safe_divide(tn, tn + fp),
        "ppv": float(precision_score(y, predictions, pos_label=1, zero_division=0)),
        "npv": safe_divide(tn, tn + fn),
        "f1": float(f1_score(y, predictions, pos_label=1, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, predictions)),
        "youden": safe_divide(tp, tp + fn) + safe_divide(tn, tn + fp) - 1.0,
        "brier": float(brier_score_loss(y, scores)),
    }
    result["auc"] = float(roc_auc_score(y, scores)) if np.unique(y).size == 2 else float("nan")
    return result


def bootstrap_confidence_intervals(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    *,
    threshold: float,
    resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap patient rows and return percentile intervals for reported metrics."""
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if len(y) != len(scores):
        raise ValueError("y_true and probabilities must have the same length.")
    generator = np.random.default_rng(seed)
    collected: dict[str, list[float]] = {metric: [] for metric in (*THRESHOLD_METRICS, "auc")}

    for _ in range(resamples):
        indices = generator.integers(0, len(y), len(y))
        sampled_y = y[indices]
        sampled_scores = scores[indices]
        values = binary_metrics(sampled_y, sampled_scores, threshold=threshold)
        for metric in THRESHOLD_METRICS:
            value = float(values[metric])
            if np.isfinite(value):
                collected[metric].append(value)
        if np.unique(sampled_y).size == 2:
            collected["auc"].append(float(values["auc"]))

    alpha = (1.0 - confidence_level) / 2.0
    point = binary_metrics(y, scores, threshold=threshold)
    rows = []
    for metric, samples in collected.items():
        if not samples:
            lower = upper = float("nan")
        else:
            lower, upper = np.quantile(samples, [alpha, 1.0 - alpha])
        rows.append(
            {
                "metric": metric,
                "estimate": float(point[metric]),
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "valid_resamples": len(samples),
            }
        )
    return pd.DataFrame(rows)


def predictions_to_metrics(
    predictions: pd.DataFrame,
    *,
    threshold: float,
    probability_column: str = "probability",
) -> dict[str, float | int]:
    return binary_metrics(
        predictions["label"].to_numpy(),
        predictions[probability_column].to_numpy(),
        threshold=threshold,
    )
