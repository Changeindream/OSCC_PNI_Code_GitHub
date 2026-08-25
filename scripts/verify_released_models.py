"""Verify released checkpoint hashes, strict loading, and optional forward passes."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from oscc_pni.app import load_registry
from oscc_pni.models.published_mil import (
    create_published_mil_model,
    forward_with_details,
    load_published_checkpoint,
)
from oscc_pni.utils import resolve_device, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("configs/model_registry.yaml"))
    parser.add_argument("--manifest", type=Path, default=Path("weights/manifest.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--forward-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    registry = load_registry(args.registry)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected = {record["file"]: record for record in manifest["weights"]}
    results = []
    for display_name, settings in registry.items():
        checkpoint = Path(settings["checkpoint"])
        record = expected[checkpoint.name]
        actual_hash = sha256_file(checkpoint)
        if actual_hash != record["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {checkpoint}")
        model = create_published_mil_model(settings["architecture"])
        metadata = load_published_checkpoint(model, checkpoint)
        result: dict[str, object] = {
            "model": display_name,
            "architecture": settings["architecture"],
            "sha256": actual_hash,
            "strict_load": True,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "epoch": metadata.get("epoch"),
        }
        if args.forward_check:
            model.to(device).eval()
            generator = torch.Generator(device=device).manual_seed(42)
            bag = torch.randn((1, 3, 224, 224), generator=generator, device=device)
            with torch.inference_mode():
                logits, attention, features = forward_with_details(model, bag)
            result.update(
                {
                    "logits_shape": list(logits.shape),
                    "attention_sum": float(attention.sum().cpu()),
                    "features_shape": list(features.shape),
                }
            )
        results.append(result)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(json.dumps({"device": str(device), "models": results}, indent=2))


if __name__ == "__main__":
    main()
