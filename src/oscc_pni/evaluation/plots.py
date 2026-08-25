"""Publication-ready plots for slice-level audits and patient-level MIL."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from oscc_pni.evaluation.metrics import binary_metrics

MODEL_COLORS = {
    "ResNet101": "#d1495b",
    "DenseNet121": "#83a744",
    "Swin Transformer": "#3f88c5",
    "Vision Transformer": "#f28e2b",
}


def _prepare_output(path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def plot_roc_curves(
    predictions: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str,
    model_column: str = "model",
) -> None:
    destination = _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(6.4, 5.5))
    for model_name, group in predictions.groupby(model_column, sort=False):
        fpr, tpr, _ = roc_curve(group["label"], group["probability"])
        auc_value = roc_auc_score(group["label"], group["probability"])
        ax.plot(
            fpr,
            tpr,
            lw=2,
            color=MODEL_COLORS.get(model_name),
            label=f"{model_name} (AUC={auc_value:.3f})",
        )
    ax.plot([0, 1], [0, 1], "--", color="0.5", lw=1, label="Random classifier")
    ax.set(
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        title=title,
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(loc="lower right", frameon=True, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_calibration_curves(
    predictions: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str = "Calibration curves",
    bins: int = 10,
) -> None:
    destination = _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(6.4, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="0.4", label="Perfect calibration")
    for model_name, group in predictions.groupby("model", sort=False):
        observed, predicted = calibration_curve(
            group["label"], group["probability"], n_bins=bins, strategy="quantile"
        )
        brier = np.mean((group["probability"].to_numpy() - group["label"].to_numpy()) ** 2)
        ax.plot(
            predicted,
            observed,
            marker="o",
            color=MODEL_COLORS.get(model_name),
            label=f"{model_name} (Brier={brier:.3f})",
        )
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Fraction of positives",
        title=title,
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination, dpi=400, bbox_inches="tight")
    plt.close(fig)


def decision_curve(
    y_true: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    n = len(y_true)
    values = []
    for threshold in thresholds:
        prediction = probabilities >= threshold
        tp = np.sum(prediction & (y_true == 1))
        fp = np.sum(prediction & (y_true == 0))
        values.append(tp / n - fp / n * threshold / (1.0 - threshold))
    return np.asarray(values)


def plot_decision_curves(predictions: pd.DataFrame, output_path: str | Path) -> None:
    destination = _prepare_output(output_path)
    thresholds = np.linspace(0.01, 0.99, 99)
    fig, ax = plt.subplots(figsize=(6.4, 5.5))
    first = next(iter(predictions.groupby("model")))[1]
    prevalence = float(first["label"].mean())
    treat_all = prevalence - (1.0 - prevalence) * thresholds / (1.0 - thresholds)
    ax.plot(thresholds, np.zeros_like(thresholds), "--", color="0.5", label="Treat none")
    ax.plot(thresholds, treat_all, "--", color="0.2", label="Treat all")
    for model_name, group in predictions.groupby("model", sort=False):
        benefit = decision_curve(
            group["label"].to_numpy(), group["probability"].to_numpy(), thresholds
        )
        ax.plot(thresholds, benefit, color=MODEL_COLORS.get(model_name), label=model_name)
    ax.set(
        xlabel="Threshold probability",
        ylabel="Net benefit",
        title="Decision-curve analysis",
        xlim=(0, 1),
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(
    predictions: pd.DataFrame,
    thresholds: dict[str, float],
    output_path: str | Path,
) -> None:
    destination = _prepare_output(output_path)
    groups = list(predictions.groupby("model", sort=False))
    columns = 2
    rows = int(np.ceil(len(groups) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(7.2, 3.4 * rows), squeeze=False)
    for axis, (model_name, group) in zip(axes.flat, groups, strict=False):
        threshold = thresholds[model_name]
        matrix = confusion_matrix(group["label"], group["probability"] >= threshold, labels=[0, 1])
        percentages = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        annotations = np.asarray(
            [[f"{matrix[i, j]}\n({percentages[i, j]:.1%})" for j in range(2)] for i in range(2)]
        )
        sns.heatmap(matrix, annot=annotations, fmt="", cmap="Blues", cbar=False, ax=axis)
        axis.set(title=model_name, xlabel="Predicted label", ylabel="True label")
        axis.set_xticklabels(["non-PNI", "PNI"])
        axis.set_yticklabels(["non-PNI", "PNI"], rotation=0)
    for axis in axes.flat[len(groups) :]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(destination, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_roi_quartile_accuracy(predictions: pd.DataFrame, output_path: str | Path) -> None:
    if "roi_area" not in predictions:
        raise ValueError("ROI-quartile analysis requires an roi_area column.")
    destination = _prepare_output(output_path)
    rows = []
    for model_name, group in predictions.groupby("model", sort=False):
        quartile = pd.qcut(
            group["roi_area"], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
        )
        for name, subset in group.assign(quartile=quartile).groupby("quartile", observed=False):
            rows.append(
                {
                    "model": model_name,
                    "quartile": str(name),
                    "accuracy": float((subset["prediction"] == subset["label"]).mean()),
                }
            )
    plot_frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    sns.barplot(data=plot_frame, x="model", y="accuracy", hue="quartile", ax=ax)
    ax.set(xlabel="", ylabel="Accuracy", title="Accuracy by ROI-size quartile", ylim=(0, 1))
    ax.legend(title="ROI quartile")
    fig.tight_layout()
    fig.savefig(destination, dpi=400, bbox_inches="tight")
    plt.close(fig)


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    z = norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    half_width = (
        z * np.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    )
    return float(center - half_width), float(center + half_width)


def plot_mil_metric_bars(
    predictions: pd.DataFrame,
    thresholds: dict[str, float],
    output_path: str | Path,
    metrics: Iterable[str] = ("mcc", "ppv", "npv", "specificity", "youden"),
) -> None:
    destination = _prepare_output(output_path)
    rows = []
    for model_name, group in predictions.groupby("model", sort=False):
        values = binary_metrics(
            group["label"], group["probability"], threshold=thresholds[model_name]
        )
        rows.extend(
            {"model": model_name, "metric": metric.upper(), "value": values[metric]}
            for metric in metrics
        )
    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    sns.barplot(data=frame, x="metric", y="value", hue="model", ax=ax)
    ax.set(xlabel="", ylabel="Value", title="Patient-level MIL performance", ylim=(0, 1.08))
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(destination, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_t_stage_accuracy(
    predictions: pd.DataFrame,
    thresholds: dict[str, float],
    output_path: str | Path,
) -> None:
    if "t_stage" not in predictions:
        raise ValueError("T-stage analysis requires a t_stage column.")
    destination = _prepare_output(output_path)
    rows = []
    for (model_name, stage), group in predictions.groupby(["model", "t_stage"], sort=False):
        correct = int(
            ((group["probability"] >= thresholds[model_name]).astype(int) == group["label"]).sum()
        )
        lower, upper = wilson_interval(correct, len(group))
        rows.append(
            {
                "model": model_name,
                "t_stage": stage,
                "accuracy": correct / len(group),
                "lower": lower,
                "upper": upper,
                "n": len(group),
            }
        )
    frame = pd.DataFrame(rows)
    stages = list(dict.fromkeys(frame["t_stage"].astype(str)))
    models = list(dict.fromkeys(frame["model"]))
    x = np.arange(len(stages))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for index, model_name in enumerate(models):
        subset = (
            frame[frame["model"] == model_name]
            .set_index(frame[frame["model"] == model_name]["t_stage"].astype(str))
            .reindex(stages)
        )
        positions = x - 0.4 + width / 2 + index * width
        values = subset["accuracy"].to_numpy(float)
        errors = np.vstack(
            [values - subset["lower"].to_numpy(float), subset["upper"].to_numpy(float) - values]
        )
        ax.bar(
            positions,
            values,
            width,
            yerr=errors,
            capsize=3,
            label=model_name,
            color=MODEL_COLORS.get(model_name),
        )
    ax.set_xticks(x, stages)
    ax.set(
        xlabel="T stage",
        ylabel="Accuracy",
        title="T-stage-specific patient-level accuracy",
        ylim=(0, 1.05),
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(destination, dpi=400, bbox_inches="tight")
    plt.close(fig)
