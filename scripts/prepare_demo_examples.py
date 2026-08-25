"""Create anonymous patient-bag examples for the public Gradio demo.

The source mapping is intentionally never written to disk. Output manifests contain
only newly assigned case IDs, slice counts, and checksums.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

LABELS = {
    "pni": {"output": "pni", "prefix": "PNI", "class_index": 1},
    "n_pni": {"output": "non_pni", "prefix": "NPNI", "class_index": 0},
}
SLICE_SUFFIX = re.compile(r"_(\d+)$")
RECORD_NUMBER = re.compile(r"\s*\(\s*\d+\s*\)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patients-per-class", type=int, default=5)
    parser.add_argument("--minimum-slices", type=int, default=3)
    parser.add_argument("--maximum-slices", type=int, default=20)
    parser.add_argument(
        "--review-dir",
        type=Path,
        help="Optional private directory for anonymous visual-review contact sheets.",
    )
    return parser.parse_args()


def _patient_key(path: Path) -> str:
    base = SLICE_SUFFIX.sub("", path.stem)
    base = RECORD_NUMBER.sub("", base)
    return " ".join(base.split()).casefold()


def _slice_number(path: Path) -> int:
    match = SLICE_SUFFIX.search(path.stem)
    return int(match.group(1)) if match else 10**12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _write_contact_sheet(case_rows: list[tuple[str, list[Path]]], output: Path) -> None:
    tile = 128
    label_width = 100
    row_height = tile + 16
    columns = max(len(paths) for _, paths in case_rows)
    sheet = Image.new("RGB", (label_width + columns * tile, len(case_rows) * row_height), "white")
    draw = ImageDraw.Draw(sheet)
    for row, (case_id, paths) in enumerate(case_rows):
        y = row * row_height
        draw.text((8, y + 52), case_id, fill="black", font=_font(16))
        for column, path in enumerate(paths):
            with Image.open(path) as image:
                tile_image = image.convert("RGB")
            sheet.paste(tile_image, (label_width + column * tile, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG", optimize=True)


def _validate_destination(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    if args.patients_per_class < 1:
        raise ValueError("--patients-per-class must be positive")
    if not 1 <= args.minimum_slices <= args.maximum_slices:
        raise ValueError("Slice limits are inconsistent")

    source = args.source.resolve()
    output = args.output.resolve()
    _validate_destination(output)

    manifest: dict[str, object] = {
        "schema_version": 1,
        "purpose": "De-identified example bags for research-software demonstration only.",
        "selection": {
            "method": "cryptographically random patient sampling within class",
            "patients_per_class": args.patients_per_class,
            "minimum_slices": args.minimum_slices,
            "maximum_slices": args.maximum_slices,
            "seed_published": False,
        },
        "deidentification": {
            "original_filenames_retained": False,
            "source_to_alias_mapping_retained": False,
            "images_reencoded": True,
            "embedded_metadata_retained": False,
            "burned_in_identifier_review": {
                "required": True,
                "status": "pending",
            },
        },
        "classes": [],
    }

    random_source = secrets.SystemRandom()
    review_rows: dict[str, list[tuple[str, list[Path]]]] = defaultdict(list)

    for source_label, settings in LABELS.items():
        grouped: dict[str, list[Path]] = defaultdict(list)
        label_directory = source / source_label
        for path in label_directory.glob("*.png"):
            grouped[_patient_key(path)].append(path)

        candidates = sorted(
            key for key, paths in grouped.items() if len(paths) >= args.minimum_slices
        )
        if len(candidates) < args.patients_per_class:
            raise ValueError(
                f"Not enough {source_label} cases with at least {args.minimum_slices} slices"
            )
        selected = random_source.sample(candidates, args.patients_per_class)

        class_record: dict[str, object] = {
            "name": "PNI" if settings["class_index"] == 1 else "Non-PNI",
            "label": settings["class_index"],
            "cases": [],
        }
        class_directory = output / str(settings["output"])

        for case_number, patient_key in enumerate(selected, start=1):
            case_id = f"{settings['prefix']}_{case_number:03d}"
            case_directory = class_directory / case_id
            case_directory.mkdir(parents=True)
            source_paths = sorted(
                grouped[patient_key], key=lambda path: (_slice_number(path), path.name.casefold())
            )[: args.maximum_slices]

            file_records: list[dict[str, object]] = []
            output_paths: list[Path] = []
            for slice_number, source_path in enumerate(source_paths, start=1):
                output_path = case_directory / f"slice_{slice_number:03d}.png"
                with Image.open(source_path) as source_image:
                    source_image.load()
                    mode = "L" if source_image.mode in {"1", "L", "I", "I;16", "F"} else "RGB"
                    anonymous_image = source_image.convert(mode)
                    anonymous_image.save(output_path, format="PNG", optimize=True)

                with Image.open(output_path) as check:
                    if check.info:
                        raise RuntimeError(f"Metadata remained in {output_path}")
                    dimensions = [check.width, check.height]
                    saved_mode = check.mode

                relative_path = output_path.relative_to(output).as_posix()
                file_records.append(
                    {
                        "path": relative_path,
                        "sha256": _sha256(output_path),
                        "dimensions": dimensions,
                        "mode": saved_mode,
                    }
                )
                output_paths.append(output_path)

            case_record = {
                "case_id": case_id,
                "slice_count": len(output_paths),
                "files": file_records,
            }
            class_record["cases"].append(case_record)
            review_rows[source_label].append((case_id, output_paths))
            print(f"Prepared {case_id}: {len(output_paths)} slices")

        manifest["classes"].append(class_record)

    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if args.review_dir:
        review_directory = args.review_dir.resolve()
        for source_label, rows in review_rows.items():
            _write_contact_sheet(rows, review_directory / f"{source_label}_contact_sheet.png")


if __name__ == "__main__":
    main()
