"""Command-line interface for the public research pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _add_common_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--backbone", required=True, choices=["resnet101", "densenet121", "vit_base", "swin_base"]
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", help="For example: cpu, cuda, or cuda:1")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oscc-pni",
        description="Paper-aligned OSCC PNI research pipeline",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    audit = subcommands.add_parser(
        "audit-manifest", help="Check privacy, labels, files, and patient leakage"
    )
    audit.add_argument("--manifest", required=True, type=Path)
    audit.add_argument("--skip-file-check", action="store_true")

    split = subcommands.add_parser(
        "split-manifest", help="Assign 7:2:1 patient-level splits by center and label"
    )
    split.add_argument("--manifest-in", required=True, type=Path)
    split.add_argument("--output", required=True, type=Path)
    split.add_argument("--train-ratio", type=float, default=0.7)
    split.add_argument("--validation-ratio", type=float, default=0.2)
    split.add_argument("--test-ratio", type=float, default=0.1)
    split.add_argument("--seed", type=int, default=42)

    prepare = subcommands.add_parser(
        "prepare-slices", help="Export tumor-bearing slices from paired NIfTI volumes"
    )
    prepare.add_argument("--images-dir", required=True, type=Path)
    prepare.add_argument("--masks-dir", required=True, type=Path)
    prepare.add_argument(
        "--id-map",
        required=True,
        type=Path,
        help="CSV with source_stem and de-identified patient_id",
    )
    prepare.add_argument("--label", required=True, type=int, choices=[0, 1])
    prepare.add_argument("--split", required=True, choices=["train", "validation", "test"])
    prepare.add_argument("--center")
    prepare.add_argument("--output-dir", required=True, type=Path)
    prepare.add_argument("--manifest-out", required=True, type=Path)
    prepare.add_argument("--append", action="store_true")
    prepare.add_argument("--apply-mask", action="store_true")
    prepare.add_argument("--padding", type=int, default=8)

    radiomics = subcommands.add_parser(
        "extract-radiomics", help="Extract reported PyRadiomics features"
    )
    radiomics.add_argument("--images-dir", required=True, type=Path)
    radiomics.add_argument("--masks-dir", required=True, type=Path)
    radiomics.add_argument("--id-map", required=True, type=Path)
    radiomics.add_argument("--label-map", type=Path, help="Optional CSV with source_stem,label")
    radiomics.add_argument("--output", required=True, type=Path)

    deep = subcommands.add_parser(
        "extract-deep-features", help="Extract frozen 3-D ResNet-18 features"
    )
    deep.add_argument("--images-dir", required=True, type=Path)
    deep.add_argument("--masks-dir", required=True, type=Path)
    deep.add_argument("--id-map", required=True, type=Path)
    deep.add_argument("--label-map", type=Path)
    deep.add_argument("--output", required=True, type=Path)
    deep.add_argument("--device")

    baselines = subcommands.add_parser(
        "train-baselines", help="Tune the exploratory conventional classifiers"
    )
    baselines.add_argument("--train", required=True, type=Path)
    baselines.add_argument("--validation", required=True, type=Path)
    baselines.add_argument("--output-dir", required=True, type=Path)
    baselines.add_argument("--iterations", type=int, default=40)
    baselines.add_argument("--folds", type=int, default=5)
    baselines.add_argument("--seed", type=int, default=42)

    train_slice = subcommands.add_parser(
        "train-slice", help="Train one reported ROI-guided single-slice model"
    )
    _add_common_training_arguments(train_slice)

    train_mil = subcommands.add_parser("train-mil", help="Train patient-level attention MIL")
    _add_common_training_arguments(train_mil)
    train_mil.add_argument("--slice-checkpoint", required=True, type=Path)

    evaluate = subcommands.add_parser("evaluate-mil", help="Evaluate a locked MIL checkpoint")
    _add_common_training_arguments(evaluate)
    evaluate.add_argument("--checkpoint", required=True, type=Path)
    evaluate.add_argument("--split", choices=["validation", "test"], default="test")
    evaluate.add_argument(
        "--threshold", type=float, help="Override only for a documented locked threshold"
    )

    explain = subcommands.add_parser(
        "explain-image", help="Generate the reported attribution maps for one image"
    )
    explain.add_argument("--image", required=True, type=Path)
    explain.add_argument(
        "--backbone", required=True, choices=["resnet101", "densenet121", "vit_base", "swin_base"]
    )
    explain.add_argument("--checkpoint", required=True, type=Path)
    explain.add_argument("--output-dir", required=True, type=Path)
    explain.add_argument("--target-class", type=int, default=1, choices=[0, 1])
    explain.add_argument(
        "--mil", action="store_true", help="Checkpoint contains an AttentionMIL model"
    )
    explain.add_argument(
        "--background-dir", type=Path, help="Training-image directory for Gradient SHAP"
    )
    explain.add_argument("--background-samples", type=int, default=16)
    explain.add_argument("--device")

    app = subcommands.add_parser("app", help="Launch the local Gradio research prototype")
    app.add_argument("--registry", required=True, type=Path)
    app.add_argument("--device")
    app.add_argument(
        "--share",
        action="store_true",
        help="Create a temporary public Gradio link; never use with identifiable images",
    )
    return parser


def _read_mapping(path: Path, value_column: str) -> dict[str, object]:
    frame = pd.read_csv(path)
    required = {"source_stem", value_column}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} must contain {sorted(required)}")
    if frame["source_stem"].duplicated().any():
        raise ValueError(f"Duplicate source_stem values in {path}")
    return dict(zip(frame["source_stem"].astype(str), frame[value_column], strict=False))


def _write_table(frame: pd.DataFrame, destination: Path, *, append: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not append:
            raise FileExistsError(
                f"Refusing to overwrite {destination}; pass --append where supported."
            )
        existing = pd.read_csv(destination)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(destination, index=False)


def _command_prepare(args: argparse.Namespace) -> None:
    from oscc_pni.data.preprocessing import export_tumor_slices, load_id_map

    frame = export_tumor_slices(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        output_dir=args.output_dir,
        id_map=load_id_map(args.id_map),
        label=args.label,
        split=args.split,
        center=args.center,
        padding=args.padding,
        apply_mask=args.apply_mask,
    )
    _write_table(frame, args.manifest_out, append=args.append)
    print(f"Wrote {len(frame)} slices from {frame['patient_id'].nunique()} patients.")


def _command_extract_radiomics(args: argparse.Namespace) -> None:
    from oscc_pni.data.preprocessing import load_id_map
    from oscc_pni.features.radiomics import extract_radiomics_directory

    label_map = _read_mapping(args.label_map, "label") if args.label_map else None
    frame = extract_radiomics_directory(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        id_map=load_id_map(args.id_map),
        label_map=label_map,
    )
    _write_table(frame, args.output)
    print(
        f"Wrote {len(frame)} patient feature rows with {len(frame.columns) - 1} extracted fields."
    )


def _command_extract_deep(args: argparse.Namespace) -> None:
    from oscc_pni.data.preprocessing import load_id_map
    from oscc_pni.features.deep_features import extract_deep_features_directory

    label_map = _read_mapping(args.label_map, "label") if args.label_map else None
    frame = extract_deep_features_directory(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        id_map=load_id_map(args.id_map),
        label_map=label_map,
        device=args.device,
    )
    _write_table(frame, args.output)
    print(f"Wrote {len(frame)} patient rows with 512 deep features.")


def _command_explain(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    from PIL import Image

    from oscc_pni.data.datasets import build_transforms
    from oscc_pni.explainability.attribution import (
        gradient_shap,
        guided_gradcam,
        integrated_gradients,
        normalize_attribution,
        occlusion_sensitivity,
        overlay_heatmap,
        smooth_gradcam_pp,
    )
    from oscc_pni.models.backbones import create_slice_model, load_checkpoint
    from oscc_pni.models.mil import create_mil_model
    from oscc_pni.utils import resolve_device

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = (
        create_mil_model(args.backbone, pretrained=False)
        if args.mil
        else create_slice_model(args.backbone, pretrained=False)
    )
    load_checkpoint(model, checkpoint, strict=True)
    model.to(device).eval()
    with Image.open(args.image) as opened:
        original = opened.convert("RGB")
    tensor = build_transforms(train=False)(original).unsqueeze(0).to(device)
    mil_proxy = bool(args.mil)
    cam = smooth_gradcam_pp(
        model,
        tensor,
        backbone=args.backbone,
        target_class=args.target_class,
        mil_single_instance=mil_proxy,
    )[0]
    guided = guided_gradcam(
        model,
        tensor,
        backbone=args.backbone,
        target_class=args.target_class,
        mil_single_instance=mil_proxy,
    )
    ig = integrated_gradients(
        model, tensor, target_class=args.target_class, steps=400, mil_single_instance=mil_proxy
    )
    occ = occlusion_sensitivity(
        model, tensor, target_class=args.target_class, mil_single_instance=mil_proxy
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_heatmap(original, cam).save(args.output_dir / "smooth_gradcam_pp.png")
    overlay_heatmap(original, normalize_attribution(guided.squeeze(0))).save(
        args.output_dir / "guided_gradcam.png"
    )
    overlay_heatmap(original, normalize_attribution(ig.squeeze(0))).save(
        args.output_dir / "integrated_gradients.png"
    )
    overlay_heatmap(original, normalize_attribution(occ.squeeze(0))).save(
        args.output_dir / "occlusion.png"
    )
    np.save(args.output_dir / "integrated_gradients.npy", ig.detach().cpu().numpy())
    np.save(args.output_dir / "occlusion.npy", occ.detach().cpu().numpy())
    if args.background_dir:
        background_paths = sorted(
            path
            for path in args.background_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )[: args.background_samples]
        if not background_paths:
            raise FileNotFoundError(f"No background images found in {args.background_dir}")
        background_tensors = []
        for path in background_paths:
            with Image.open(path) as opened:
                background_tensors.append(build_transforms(train=False)(opened.convert("RGB")))
        background = torch.stack(background_tensors).to(device)
        shap_values = gradient_shap(
            model,
            tensor,
            background,
            target_class=args.target_class,
            mil_single_instance=mil_proxy,
        )
        overlay_heatmap(original, normalize_attribution(shap_values.squeeze(0))).save(
            args.output_dir / "gradient_shap.png"
        )
        np.save(args.output_dir / "gradient_shap.npy", shap_values.detach().cpu().numpy())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit-manifest":
        from oscc_pni.privacy import audit_manifest

        report = audit_manifest(args.manifest, check_files=not args.skip_file_check)
        print(report.format())
        return 0 if report.ok else 2
    if args.command == "split-manifest":
        from oscc_pni.data.manifest import stratified_patient_split

        frame = pd.read_csv(args.manifest_in)
        output = stratified_patient_split(
            frame,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        _write_table(output, args.output)
        print(output.drop_duplicates("patient_id").groupby(["split", "label"]).size())
        return 0
    if args.command == "prepare-slices":
        _command_prepare(args)
    elif args.command == "extract-radiomics":
        _command_extract_radiomics(args)
    elif args.command == "extract-deep-features":
        _command_extract_deep(args)
    elif args.command == "train-baselines":
        from oscc_pni.features.classical import fit_exploratory_models, save_baseline_results

        results = fit_exploratory_models(
            pd.read_csv(args.train),
            pd.read_csv(args.validation),
            seed=args.seed,
            iterations=args.iterations,
            folds=args.folds,
        )
        save_baseline_results(results, args.output_dir)
        print(json.dumps({result.name: result.validation_auc for result in results}, indent=2))
    elif args.command == "train-slice":
        from oscc_pni.training.engine import train_slice_model

        path = train_slice_model(
            manifest_path=args.manifest,
            backbone=args.backbone,
            output_dir=args.output_dir,
            config_path=args.config,
            device_name=args.device,
        )
        print(path)
    elif args.command == "train-mil":
        from oscc_pni.training.engine import train_mil_model

        path = train_mil_model(
            manifest_path=args.manifest,
            backbone=args.backbone,
            slice_checkpoint=args.slice_checkpoint,
            output_dir=args.output_dir,
            config_path=args.config,
            device_name=args.device,
        )
        print(path)
    elif args.command == "evaluate-mil":
        from oscc_pni.training.engine import evaluate_mil_checkpoint

        metrics = evaluate_mil_checkpoint(
            manifest_path=args.manifest,
            backbone=args.backbone,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            split=args.split,
            threshold=args.threshold,
            config_path=args.config,
            device_name=args.device,
        )
        print(json.dumps(metrics, indent=2))
    elif args.command == "explain-image":
        _command_explain(args)
    elif args.command == "app":
        from oscc_pni.app import launch

        launch(args.registry, device_name=args.device, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
