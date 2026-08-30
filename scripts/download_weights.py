"""Download the published MIL checkpoints from Hugging Face and verify them."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

BUFFER_BYTES = 8 * 1024 * 1024
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "weights" / "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(BUFFER_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("storage") != "huggingface" or not isinstance(
        manifest.get("weights"), list
    ):
        raise ValueError("Expected a Hugging Face weight manifest.")
    return manifest


def is_valid(path: Path, record: dict[str, Any]) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == int(record["bytes"])
        and sha256(path) == str(record["sha256"])
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--file", action="append", dest="files")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_manifest(manifest_path)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else manifest_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    records = {str(record["file"]): record for record in manifest["weights"]}
    requested = args.files or list(records)
    missing = sorted(set(requested) - set(records))
    if missing:
        raise ValueError(f"Files absent from the manifest: {', '.join(missing)}")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "Install the app dependencies first: "
            "python -m pip install -r requirements-app-cu128.txt"
        ) from exc

    for filename in requested:
        record = records[filename]
        destination = output_dir / filename
        if is_valid(destination, record):
            print(f"Already valid: {destination}")
            continue
        if destination.exists() and not args.force:
            raise ValueError(
                f"Refusing to overwrite an invalid checkpoint: {destination}. "
                "Review it, then rerun with --force if replacement is intended."
            )
        if destination.exists():
            destination.unlink()

        downloaded = Path(
            hf_hub_download(
                repo_id=str(manifest["repository_id"]),
                filename=filename,
                revision=str(manifest["revision"]),
                repo_type="model",
                local_dir=output_dir,
                force_download=args.force,
            )
        )
        if downloaded.resolve() != destination.resolve() and not destination.is_file():
            raise RuntimeError(f"Unexpected download location: {downloaded}")
        if not is_valid(destination, record):
            raise ValueError(f"Downloaded checkpoint failed integrity validation: {destination}")
        print(f"Downloaded and verified: {destination}")


if __name__ == "__main__":
    main()
