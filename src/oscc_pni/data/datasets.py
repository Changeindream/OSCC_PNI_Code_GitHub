"""Single-slice and patient-bag PyTorch datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from oscc_pni.utils import natural_key

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(
    *,
    train: bool,
    image_size: int = 224,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> transforms.Compose:
    """Construct the augmentation family described in the supplementary methods."""
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.80, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.RandomAffine(0, translate=(0.10, 0.10), scale=(0.90, 1.10)),
                transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.10),
                transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.30),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.RandomErasing(p=0.30, scale=(0.02, 0.15)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def _load_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


class SingleSliceDataset(Dataset[dict[str, Any]]):
    """Manifest-backed ROI-slice dataset without filename-derived labels."""

    def __init__(self, frame: pd.DataFrame, transform: Any = None) -> None:
        if frame.empty:
            raise ValueError("The single-slice dataset is empty.")
        self.frame = frame.reset_index(drop=True).copy()
        self.transform = transform or build_transforms(train=False)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image = self.transform(_load_rgb(row["image_path"]))
        item: dict[str, Any] = {
            "image": image,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "patient_id": str(row["patient_id"]),
            "image_path": str(row["image_path"]),
        }
        for column in ("center", "t_stage", "roi_area", "slice_index"):
            if column in row and pd.notna(row[column]):
                item[column] = row[column]
        return item


class PatientBagDataset(Dataset[dict[str, Any]]):
    """Return all tumor-bearing slices from one patient as one MIL bag.

    Training bags can be stochastically subsampled without replacement. Evaluation
    bags are complete and naturally ordered for deterministic patient-level output.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        transform: Any = None,
        *,
        training: bool = False,
        max_instances: int | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        if frame.empty:
            raise ValueError("The patient-bag dataset is empty.")
        self.transform = transform or build_transforms(train=training)
        self.training = training
        self.max_instances = max_instances if training else None
        self.generator = generator
        self._bags: list[pd.DataFrame] = []

        for patient_id, group in frame.groupby("patient_id", sort=True):
            labels = group["label"].unique()
            if len(labels) != 1:
                raise ValueError(f"Patient {patient_id} has conflicting labels: {labels.tolist()}")
            ordered = group.copy()
            if "slice_index" in ordered and ordered["slice_index"].notna().all():
                ordered = ordered.sort_values("slice_index")
            else:
                order = sorted(
                    range(len(ordered)),
                    key=lambda i: natural_key(str(ordered.iloc[i]["image_path"])),
                )
                ordered = ordered.iloc[order]
            self._bags.append(ordered.reset_index(drop=True))

    def __len__(self) -> int:
        return len(self._bags)

    def __getitem__(self, index: int) -> dict[str, Any]:
        bag_frame = self._bags[index]
        selected = bag_frame
        if self.max_instances and len(bag_frame) > self.max_instances:
            indices = torch.randperm(len(bag_frame), generator=self.generator)[: self.max_instances]
            selected = bag_frame.iloc[indices.sort().values.tolist()]

        images = [self.transform(_load_rgb(path)) for path in selected["image_path"]]
        if not images:
            raise RuntimeError(f"No readable slices for patient {bag_frame.iloc[0]['patient_id']}")

        item: dict[str, Any] = {
            "images": torch.stack(images),
            "label": torch.tensor(int(bag_frame.iloc[0]["label"]), dtype=torch.long),
            "patient_id": str(bag_frame.iloc[0]["patient_id"]),
            "image_paths": selected["image_path"].astype(str).tolist(),
            "num_instances": len(selected),
        }
        for column in ("center", "t_stage"):
            if column in bag_frame and pd.notna(bag_frame.iloc[0][column]):
                item[column] = bag_frame.iloc[0][column]
        return item


def single_patient_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Explicitly enforce batch size one for variable-length patient bags."""
    if len(batch) != 1:
        raise ValueError("MIL uses batch_size=1 because patient bags have variable length.")
    return batch[0]
