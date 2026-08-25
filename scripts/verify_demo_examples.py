"""Verify the privacy-oriented naming, metadata removal, and integrity of demo images."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / "examples"
MANIFEST = ROOT / "manifest.json"
CASE_ID = re.compile(r"(?:PNI|NPNI)_\d{3}")
SLICE_NAME = re.compile(r"slice_\d{3}\.png")
SUSPICIOUS = re.compile(r"\(\s*\d{5,}\s*\)|(?<!\d)\d{7,}(?!\d)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path = ROOT, manifest_path: Path = MANIFEST) -> tuple[int, int]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_paths: set[Path] = set()
    case_count = 0

    for class_record in manifest["classes"]:
        for case in class_record["cases"]:
            case_count += 1
            case_id = str(case["case_id"])
            if not CASE_ID.fullmatch(case_id):
                raise ValueError(f"Unsafe case ID: {case_id}")
            if int(case["slice_count"]) != len(case["files"]):
                raise ValueError(f"Slice-count mismatch for {case_id}")

            for record in case["files"]:
                relative = Path(str(record["path"]))
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Unsafe relative path: {relative}")
                if not SLICE_NAME.fullmatch(relative.name) or SUSPICIOUS.search(relative.as_posix()):
                    raise ValueError(f"Potential identifier in path: {relative}")
                path = (root / relative).resolve()
                if root.resolve() not in path.parents:
                    raise ValueError(f"Path escapes example root: {relative}")
                if not path.is_file():
                    raise FileNotFoundError(path)
                if _sha256(path) != record["sha256"]:
                    raise ValueError(f"Checksum mismatch: {relative}")

                with Image.open(path) as image:
                    if image.info:
                        raise ValueError(f"Embedded metadata detected: {relative}")
                    if [image.width, image.height] != record["dimensions"]:
                        raise ValueError(f"Dimension mismatch: {relative}")
                    if image.mode != record["mode"]:
                        raise ValueError(f"Mode mismatch: {relative}")
                expected_paths.add(path)

    actual_paths = {path.resolve() for path in root.rglob("*.png")}
    extras = actual_paths - expected_paths
    missing = expected_paths - actual_paths
    if extras or missing:
        raise ValueError(f"Manifest/file mismatch: extras={len(extras)}, missing={len(missing)}")
    return case_count, len(expected_paths)


if __name__ == "__main__":
    cases, images = verify()
    print(f"Demo example audit: PASS ({cases} cases, {images} images)")
