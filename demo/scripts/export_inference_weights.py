from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an inference-only PyTorch state dictionary.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False, mmap=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("No model state dictionary was found in the checkpoint.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, args.output)

    entry = {
        "model": args.model,
        "file": args.output.name,
        "source_file": args.input.name,
        "source_sha256": sha256(args.input),
        "inference_sha256": sha256(args.output),
        "state_entries": len(state_dict),
        "parameters_and_buffers": sum(value.numel() for value in state_dict.values()),
        "bytes": args.output.stat().st_size,
    }
    manifest = {"format": "PyTorch state_dict", "models": []}
    if args.manifest.exists():
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest["created_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["models"] = [item for item in manifest.get("models", []) if item["model"] != args.model]
    manifest["models"].append(entry)
    manifest["models"].sort(key=lambda item: item["model"])
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()

