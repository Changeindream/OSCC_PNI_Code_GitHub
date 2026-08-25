"""Convert trusted training checkpoints into compact, weights-only inference files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

FILENAMES = {
    "resnet152_mil": "resnet152_mil_best_auc.pt",
    "densenet121_mil": "densenet121_mil_best_auc.pt",
    "swin_base_mil": "swin_base_mil_best_auc.pt",
    "vit_base_mil": "vit_base_mil_best_auc.pt",
}


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_checkpoint(architecture: str, source: Path, destination: Path) -> dict[str, object]:
    if not source.is_file():
        raise FileNotFoundError(source)
    # The source checkpoints were produced locally and are trusted.  Released
    # checkpoints are saved with tensors and primitive metadata only, allowing
    # downstream loading with weights_only=True.
    try:
        checkpoint = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError:
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
        raise TypeError(f"Unsupported training checkpoint format: {source}")
    payload = {
        "format_version": 1,
        "architecture": architecture,
        "model_state_dict": checkpoint["model_state_dict"],
        "epoch": int(checkpoint["epoch"]) if "epoch" in checkpoint else None,
        "best_auc": float(checkpoint["best_auc"]) if "best_auc" in checkpoint else None,
        "source_sha256": sha256(source),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    return {
        "architecture": architecture,
        "file": destination.name,
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "source_bytes": source.stat().st_size,
        "source_sha256": payload["source_sha256"],
        "epoch": payload["epoch"],
        "best_auc": payload["best_auc"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    for architecture in FILENAMES:
        parser.add_argument(f"--{architecture.replace('_', '-')}", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    for architecture, filename in FILENAMES.items():
        source = getattr(args, architecture)
        records.append(export_checkpoint(architecture, source, args.output_dir / filename))
    manifest = {"format_version": 1, "weights": records}
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
