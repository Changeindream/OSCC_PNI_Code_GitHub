"""Split, reconstruct, and verify byte-identical released checkpoint parts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import BinaryIO

BUFFER_BYTES = 4 * 1024 * 1024


def _sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(BUFFER_BYTES), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return _sha256_stream(stream)


def _load_manifest(path: Path) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest.get("weights"), list):
        raise ValueError("The weight manifest does not contain a weights list.")
    return manifest


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _selected_records(
    manifest: dict[str, object], requested: list[str] | None
) -> list[dict[str, object]]:
    records = [record for record in manifest["weights"] if isinstance(record, dict)]
    if not requested:
        return records
    by_name = {str(record["file"]): record for record in records}
    missing = sorted(set(requested) - set(by_name))
    if missing:
        raise ValueError(f"Files are absent from the manifest: {', '.join(missing)}")
    return [by_name[name] for name in requested]


def split_weights(
    manifest_path: Path,
    weights_dir: Path,
    requested: list[str],
    chunk_mib: int,
    remove_originals: bool,
) -> None:
    if chunk_mib < 1:
        raise ValueError("--chunk-mib must be positive.")
    manifest = _load_manifest(manifest_path)
    parts_dir = weights_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    chunk_bytes = chunk_mib * 1024 * 1024

    for record in _selected_records(manifest, requested):
        filename = str(record["file"])
        source = weights_dir / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.stat().st_size != int(record["bytes"]) or _sha256(source) != record["sha256"]:
            raise ValueError(f"Source checkpoint failed integrity validation: {source}")

        old_parts = list(parts_dir.glob(f"{filename}.part-*"))
        if old_parts:
            raise FileExistsError(f"Existing parts must be removed before splitting {filename}.")

        part_records: list[dict[str, object]] = []
        with source.open("rb") as input_stream:
            for number in range(1, 10000):
                payload = input_stream.read(chunk_bytes)
                if not payload:
                    break
                part_name = f"{filename}.part-{number:03d}"
                part_path = parts_dir / part_name
                part_path.write_bytes(payload)
                part_records.append(
                    {
                        "file": f"parts/{part_name}",
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )

        record["storage"] = "split-lfs"
        record["parts"] = part_records
        print(f"Split {filename} into {len(part_records)} byte-identical parts.")

    _write_manifest(manifest_path, manifest)

    if remove_originals:
        for record in _selected_records(manifest, requested):
            target = weights_dir / str(record["file"])
            if target.is_file():
                target.unlink()
                print(f"Removed split source: {target}")


def reconstruct_weights(
    manifest_path: Path,
    weights_dir: Path,
    requested: list[str] | None,
) -> None:
    manifest = _load_manifest(manifest_path)
    for record in _selected_records(manifest, requested):
        parts = record.get("parts")
        if not parts:
            continue
        target = weights_dir / str(record["file"])
        expected_size = int(record["bytes"])
        expected_hash = str(record["sha256"])
        if target.is_file():
            if target.stat().st_size == expected_size and _sha256(target) == expected_hash:
                print(f"Already valid: {target}")
                continue
            raise ValueError(f"Refusing to overwrite an invalid existing checkpoint: {target}")

        temporary = target.with_suffix(target.suffix + ".assembling")
        digest = hashlib.sha256()
        written = 0
        try:
            with temporary.open("wb") as output_stream:
                for part in parts:
                    part_path = weights_dir / str(part["file"])
                    if not part_path.is_file():
                        raise FileNotFoundError(part_path)
                    if part_path.stat().st_size != int(part["bytes"]):
                        raise ValueError(f"Part-size mismatch: {part_path}")
                    if _sha256(part_path) != part["sha256"]:
                        raise ValueError(f"Part checksum mismatch: {part_path}")
                    with part_path.open("rb") as input_stream:
                        for chunk in iter(lambda: input_stream.read(BUFFER_BYTES), b""):
                            output_stream.write(chunk)
                            digest.update(chunk)
                            written += len(chunk)
            if written != expected_size or digest.hexdigest() != expected_hash:
                raise ValueError(f"Reconstructed checkpoint failed integrity validation: {target}")
            os.replace(temporary, target)
            print(f"Reconstructed and verified: {target}")
        finally:
            if temporary.exists():
                temporary.unlink()


def verify_storage(manifest_path: Path, weights_dir: Path) -> None:
    manifest = _load_manifest(manifest_path)
    for record in _selected_records(manifest, None):
        target = weights_dir / str(record["file"])
        verified = False
        if target.is_file():
            if target.stat().st_size != int(record["bytes"]) or _sha256(target) != record["sha256"]:
                raise ValueError(f"Checkpoint checksum mismatch: {target}")
            print(f"Valid checkpoint: {target}")
            verified = True
        parts = record.get("parts")
        if not parts and not verified:
            raise FileNotFoundError(target)
        if parts:
            total = 0
            for part in parts:
                part_path = weights_dir / str(part["file"])
                if part_path.stat().st_size != int(part["bytes"]):
                    raise ValueError(f"Part-size mismatch: {part_path}")
                if _sha256(part_path) != part["sha256"]:
                    raise ValueError(f"Part checksum mismatch: {part_path}")
                total += int(part["bytes"])
            if total != int(record["bytes"]):
                raise ValueError(f"Part total does not match checkpoint size: {target.name}")
            print(f"Valid split checkpoint: {target.name} ({len(parts)} parts)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("split", "reconstruct", "verify"))
    parser.add_argument("--manifest", type=Path, default=Path("weights/manifest.json"))
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--file", action="append", dest="files")
    parser.add_argument("--chunk-mib", type=int, default=64)
    parser.add_argument("--remove-originals", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    weights_dir = args.weights_dir.resolve()
    if args.command == "split":
        if not args.files:
            raise ValueError("split requires at least one --file argument.")
        split_weights(
            manifest_path,
            weights_dir,
            args.files,
            args.chunk_mib,
            args.remove_originals,
        )
    elif args.command == "reconstruct":
        reconstruct_weights(manifest_path, weights_dir, args.files)
    else:
        verify_storage(manifest_path, weights_dir)


if __name__ == "__main__":
    main()
