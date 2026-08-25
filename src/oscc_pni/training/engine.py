"""Training and locked patient-level evaluation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from oscc_pni.config import load_config, normalize_backbone_name
from oscc_pni.data.datasets import (
    PatientBagDataset,
    SingleSliceDataset,
    build_transforms,
    single_patient_collate,
)
from oscc_pni.data.manifest import load_manifest, manifest_summary, validate_manifest
from oscc_pni.evaluation.metrics import (
    binary_metrics,
    bootstrap_confidence_intervals,
    select_threshold,
)
from oscc_pni.models.backbones import create_slice_model, load_checkpoint
from oscc_pni.models.mil import create_mil_model
from oscc_pni.training.losses import (
    CrossEntropyPlusFocal,
    FocalLoss,
    cutmix,
    mixed_loss,
    mixup,
)
from oscc_pni.utils import resolve_device, seed_everything, sha256_file, write_run_metadata


def _amp_enabled(device: torch.device, requested: bool) -> bool:
    return bool(requested and device.type == "cuda")


def _make_grad_scaler(enabled: bool):
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _slice_criterion(backbone: str) -> nn.Module:
    return CrossEntropyPlusFocal() if backbone == "vit_base" else FocalLoss(gamma=2.0)


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _history_row(
    epoch: int, train_loss: float, validation_loss: float, predictions: pd.DataFrame
) -> dict[str, float | int]:
    auc_value = roc_auc_score(predictions["label"], predictions["probability"])
    return {
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "validation_auc": float(auc_value),
    }


@torch.inference_mode()
def predict_slices(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> tuple[pd.DataFrame, float]:
    model.eval()
    rows: list[dict[str, Any]] = []
    loss_sum = 0.0
    samples = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits = model(images)
        if criterion is not None:
            loss_sum += float(criterion(logits, labels).item()) * len(labels)
        probabilities = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        for index in range(len(labels)):
            row: dict[str, Any] = {
                "patient_id": str(batch["patient_id"][index]),
                "image_path": str(batch["image_path"][index]),
                "label": int(labels[index].item()),
                "probability": float(probabilities[index]),
                "prediction": int(predictions[index]),
            }
            for column in ("center", "t_stage", "roi_area", "slice_index"):
                if column in batch:
                    value = batch[column][index]
                    row[column] = value.item() if hasattr(value, "item") else value
            rows.append(row)
        samples += len(labels)
    return pd.DataFrame(rows), loss_sum / max(samples, 1)


def train_slice_model(
    *,
    manifest_path: str | Path,
    backbone: str,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    device_name: str | None = None,
) -> Path:
    """Train one of the four reported ROI-guided single-slice classifiers."""
    config = load_config(config_path)
    backbone = normalize_backbone_name(backbone)
    seed = int(config["study"]["seed"])
    seed_everything(seed)
    device = resolve_device(device_name)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    frame = load_manifest(manifest_path)
    report = validate_manifest(frame, check_files=True)
    if not report.ok:
        raise ValueError(report.format())
    manifest_summary(frame).to_csv(output / "manifest_summary.csv", index=False)

    model_config = config["slice_models"][backbone]
    image_size = int(config["preprocessing"]["image_size"])
    train_frame = frame[frame["split"] == "train"]
    validation_frame = frame[frame["split"] == "validation"]
    train_dataset = SingleSliceDataset(
        train_frame, build_transforms(train=True, image_size=image_size)
    )
    validation_dataset = SingleSliceDataset(
        validation_frame, build_transforms(train=False, image_size=image_size)
    )
    batch_size = int(model_config["batch_size"])
    workers = int(config["training"]["num_workers"])
    loader_kwargs = {"num_workers": workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs
    )

    model = create_slice_model(
        backbone,
        dropout=float(model_config["dropout"]),
        pretrained=True,
    ).to(device)
    criterion = _slice_criterion(backbone)
    optimizer = AdamW(
        model.parameters(),
        lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    epochs = int(config["training"]["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    amp = _amp_enabled(device, bool(config["training"]["mixed_precision"]))
    scaler = _make_grad_scaler(amp)
    mixup_alpha = float(model_config.get("mixup_alpha", 0.0))
    cutmix_alpha = float(model_config.get("cutmix_alpha", 0.0))
    patience = int(config["training"]["early_stopping_patience"])
    best_auc = -math.inf
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    best_path = output / "best.pt"

    write_run_metadata(
        output,
        config,
        extra={
            "manifest_sha256": sha256_file(manifest_path),
            "backbone": backbone,
            "task": "slice",
        },
    )

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_samples = 0
        progress = tqdm(train_loader, desc=f"slice {backbone} epoch {epoch}/{epochs}", leave=False)
        for batch in progress:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            first, second, lam = labels, labels, 1.0
            draw = np.random.random()
            if backbone == "vit_base" and cutmix_alpha > 0 and draw < 0.25:
                images, first, second, lam = cutmix(images, labels, alpha=cutmix_alpha)
            elif mixup_alpha > 0 and draw < 0.50:
                images, first, second, lam = mixup(images, labels, alpha=mixup_alpha)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(images)
                loss = mixed_loss(criterion, logits, first, second, lam)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += float(loss.item()) * len(labels)
            train_samples += len(labels)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        validation_predictions, validation_loss = predict_slices(
            model, validation_loader, device, criterion
        )
        row = _history_row(
            epoch,
            train_loss_sum / max(train_samples, 1),
            validation_loss,
            validation_predictions,
        )
        history.append(row)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        validation_predictions.to_csv(output / "latest_validation_predictions.csv", index=False)

        if float(row["validation_auc"]) > best_auc:
            best_auc = float(row["validation_auc"])
            stale_epochs = 0
            threshold = select_threshold(
                validation_predictions["label"], validation_predictions["probability"]
            )
            _save_checkpoint(
                best_path,
                {
                    "model_state_dict": model.state_dict(),
                    "backbone": backbone,
                    "epoch": epoch,
                    "validation_auc": best_auc,
                    "validation_threshold": threshold,
                    "config": config,
                },
            )
            validation_predictions.to_csv(output / "best_validation_predictions.csv", index=False)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    return best_path


@torch.inference_mode()
def predict_patient_bags(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    model.eval()
    patient_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    loss_sum = 0.0
    for item in loader:
        images = item["images"].to(device, non_blocking=True)
        label = item["label"].view(1).to(device)
        logits, attention = model(images, return_attention=True)
        logits = logits.view(1, -1)
        if criterion is not None:
            loss_sum += float(criterion(logits, label).item())
        probability = float(torch.softmax(logits, dim=-1)[0, 1].item())
        row: dict[str, Any] = {
            "patient_id": str(item["patient_id"]),
            "label": int(label.item()),
            "probability": probability,
            "prediction": int(probability >= 0.5),
            "num_instances": int(item["num_instances"]),
        }
        for column in ("center", "t_stage"):
            if column in item:
                row[column] = item[column]
        patient_rows.append(row)
        for image_path, weight in zip(
            item["image_paths"], attention.detach().cpu().tolist(), strict=False
        ):
            attention_rows.append(
                {
                    "patient_id": str(item["patient_id"]),
                    "image_path": str(image_path),
                    "attention_weight": float(weight),
                }
            )
    count = max(len(patient_rows), 1)
    return pd.DataFrame(patient_rows), pd.DataFrame(attention_rows), loss_sum / count


def train_mil_model(
    *,
    manifest_path: str | Path,
    backbone: str,
    slice_checkpoint: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    device_name: str | None = None,
) -> Path:
    """Train patient-level MIL initialized from the corresponding slice model."""
    config = load_config(config_path)
    backbone = normalize_backbone_name(backbone)
    seed = int(config["study"]["seed"])
    seed_everything(seed)
    device = resolve_device(device_name)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    frame = load_manifest(manifest_path)
    report = validate_manifest(frame, check_files=True)
    if not report.ok:
        raise ValueError(report.format())
    manifest_summary(frame).to_csv(output / "manifest_summary.csv", index=False)

    image_size = int(config["preprocessing"]["image_size"])
    max_instances = int(config["mil"]["max_train_instances"])
    generator = torch.Generator().manual_seed(seed)
    train_dataset = PatientBagDataset(
        frame[frame["split"] == "train"],
        build_transforms(train=True, image_size=image_size),
        training=True,
        max_instances=max_instances,
        generator=generator,
    )
    validation_dataset = PatientBagDataset(
        frame[frame["split"] == "validation"],
        build_transforms(train=False, image_size=image_size),
        training=False,
    )
    workers = int(config["training"]["num_workers"])
    loader_kwargs = {
        "batch_size": 1,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": single_patient_collate,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)

    model = create_mil_model(
        backbone,
        attention_dim=int(config["mil"]["attention_dim"]),
        pretrained=False,
        slice_checkpoint=slice_checkpoint,
    ).to(device)
    criterion = FocalLoss(
        gamma=float(config["mil"]["focal_gamma"]),
        alpha=float(config["mil"]["focal_alpha"]),
    )
    optimizer = AdamW(
        [
            {
                "params": model.encoder.parameters(),
                "lr": float(config["mil"]["backbone_learning_rate"]),
            },
            {
                "params": list(model.attention.parameters()) + list(model.classifier.parameters()),
                "lr": float(config["mil"]["head_learning_rate"]),
            },
        ],
        weight_decay=float(config["mil"]["weight_decay"]),
    )
    epochs = int(config["training"]["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    amp = _amp_enabled(device, bool(config["training"]["mixed_precision"]))
    scaler = _make_grad_scaler(amp)
    patience = int(config["training"]["early_stopping_patience"])
    best_auc = -math.inf
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    best_path = output / "best.pt"

    write_run_metadata(
        output,
        config,
        extra={
            "manifest_sha256": sha256_file(manifest_path),
            "slice_checkpoint_sha256": sha256_file(slice_checkpoint),
            "backbone": backbone,
            "task": "mil",
        },
    )

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        progress = tqdm(train_loader, desc=f"MIL {backbone} epoch {epoch}/{epochs}", leave=False)
        for item in progress:
            images = item["images"].to(device, non_blocking=True)
            label = item["label"].view(1).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(images).view(1, -1)
                loss = criterion(logits, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += float(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()

        validation_predictions, validation_attention, validation_loss = predict_patient_bags(
            model, validation_loader, device, criterion
        )
        row = _history_row(
            epoch,
            train_loss_sum / max(len(train_loader), 1),
            validation_loss,
            validation_predictions,
        )
        history.append(row)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        validation_predictions.to_csv(output / "latest_validation_predictions.csv", index=False)

        if float(row["validation_auc"]) > best_auc:
            best_auc = float(row["validation_auc"])
            stale_epochs = 0
            threshold = select_threshold(
                validation_predictions["label"], validation_predictions["probability"]
            )
            _save_checkpoint(
                best_path,
                {
                    "model_state_dict": model.state_dict(),
                    "backbone": backbone,
                    "epoch": epoch,
                    "validation_auc": best_auc,
                    "validation_threshold": threshold,
                    "config": config,
                },
            )
            validation_predictions.to_csv(output / "best_validation_predictions.csv", index=False)
            validation_attention.to_csv(output / "best_validation_attention.csv", index=False)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    return best_path


def evaluate_mil_checkpoint(
    *,
    manifest_path: str | Path,
    backbone: str,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    split: str = "test",
    threshold: float | None = None,
    config_path: str | Path | None = None,
    device_name: str | None = None,
) -> dict[str, float | int]:
    """Evaluate a locked checkpoint once, using the saved validation threshold."""
    config = load_config(config_path)
    backbone = normalize_backbone_name(backbone)
    seed_everything(int(config["study"]["seed"]))
    device = resolve_device(device_name)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = load_manifest(manifest_path)
    report = validate_manifest(frame, check_files=True)
    if not report.ok:
        raise ValueError(report.format())
    split_frame = frame[frame["split"] == split]
    if split_frame.empty:
        raise ValueError(f"Split '{split}' is empty.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = create_mil_model(
        backbone,
        attention_dim=int(config["mil"]["attention_dim"]),
        pretrained=False,
    )
    load_checkpoint(model, checkpoint, strict=True)
    model.to(device).eval()
    dataset = PatientBagDataset(
        split_frame,
        build_transforms(train=False, image_size=int(config["preprocessing"]["image_size"])),
        training=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=single_patient_collate)
    predictions, attention, _ = predict_patient_bags(model, loader, device)

    locked_threshold = threshold
    if locked_threshold is None:
        locked_threshold = checkpoint.get("validation_threshold")
    if locked_threshold is None:
        raise ValueError("No locked validation threshold was provided or stored in the checkpoint.")
    locked_threshold = float(locked_threshold)
    predictions["prediction"] = (predictions["probability"] >= locked_threshold).astype(int)
    metrics = binary_metrics(
        predictions["label"], predictions["probability"], threshold=locked_threshold
    )
    intervals = bootstrap_confidence_intervals(
        predictions["label"],
        predictions["probability"],
        threshold=locked_threshold,
        resamples=int(config["evaluation"]["bootstrap_resamples"]),
        confidence_level=float(config["evaluation"]["confidence_level"]),
        seed=int(config["study"]["seed"]),
    )
    predictions.to_csv(output / f"{split}_predictions.csv", index=False)
    attention.to_csv(output / f"{split}_attention.csv", index=False)
    intervals.to_csv(output / f"{split}_bootstrap_intervals.csv", index=False)
    (output / f"{split}_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_run_metadata(
        output,
        config,
        extra={
            "manifest_sha256": sha256_file(manifest_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "backbone": backbone,
            "task": "mil_evaluation",
            "split": split,
            "locked_threshold": locked_threshold,
        },
    )
    return metrics
