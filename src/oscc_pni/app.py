"""Local Gradio interface for the four released patient-level MIL models."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image

from oscc_pni.data.datasets import build_transforms
from oscc_pni.explainability.attribution import overlay_heatmap, smooth_gradcam_pp
from oscc_pni.models.published_mil import (
    PUBLISHED_ARCHITECTURES,
    create_published_mil_model,
    forward_with_details,
    load_published_checkpoint,
)
from oscc_pni.utils import natural_key, resolve_device

DEFAULT_CLASS_NAMES = ("Non-PNI", "PNI")


def _read_registry(path: str | Path) -> tuple[Path, dict[str, Any]]:
    registry_path = Path(path).expanduser().resolve()
    content = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    if not isinstance(content, dict):
        raise TypeError("The model registry must contain a YAML mapping.")
    return registry_path, content


def load_registry(path: str | Path) -> dict[str, dict[str, str]]:
    """Resolve the public model registry without exposing machine-specific paths."""
    registry_path, content = _read_registry(path)
    models = content.get("models", {})
    if not isinstance(models, dict) or not models:
        raise ValueError("The registry does not define any models.")
    resolved: dict[str, dict[str, str]] = {}
    for display_name, settings in models.items():
        if not isinstance(settings, dict):
            raise TypeError(f"Settings for {display_name!r} must be a mapping.")
        architecture = str(settings["architecture"])
        if architecture not in PUBLISHED_ARCHITECTURES:
            raise ValueError(f"Unsupported published architecture: {architecture}")
        checkpoint = Path(settings["checkpoint"]).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = (registry_path.parent / checkpoint).resolve()
        resolved[str(display_name)] = {
            "architecture": architecture,
            "checkpoint": str(checkpoint),
        }
    return resolved


def load_inference_settings(path: str | Path) -> dict[str, Any]:
    """Load UI-only settings while applying safe, paper-aligned defaults."""
    _, content = _read_registry(path)
    settings = content.get("inference", {}) or {}
    class_names = tuple(str(value) for value in settings.get("class_names", DEFAULT_CLASS_NAMES))
    if len(class_names) != 2:
        raise ValueError("Exactly two class names are required.")
    max_slices = int(settings.get("max_slices_per_patient", 20))
    if max_slices < 1:
        raise ValueError("max_slices_per_patient must be positive.")
    return {
        "class_names": class_names,
        "image_size": int(settings.get("image_size", 224)),
        "max_slices_per_patient": max_slices,
    }


class ModelService:
    """Lazy model loading and deterministic single-patient inference."""

    def __init__(
        self,
        registry: dict[str, dict[str, str]],
        *,
        device_name: str | None = None,
        class_names: tuple[str, str] = DEFAULT_CLASS_NAMES,
        image_size: int = 224,
        max_slices_per_patient: int = 20,
    ) -> None:
        self.registry = registry
        self.device = resolve_device(device_name)
        self.class_names = class_names
        self.max_slices_per_patient = max_slices_per_patient
        self.cache: dict[str, torch.nn.Module] = {}
        self.metadata: dict[str, dict[str, Any]] = {}
        self.transform = build_transforms(train=False, image_size=image_size)

    def load(self, display_name: str) -> torch.nn.Module:
        if display_name in self.cache:
            return self.cache[display_name]
        settings = self.registry[display_name]
        checkpoint_path = Path(settings["checkpoint"])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. Run "
                "'python scripts/download_weights.py' from the repository root."
            )
        model = create_published_mil_model(settings["architecture"])
        metadata = load_published_checkpoint(model, checkpoint_path)
        checkpoint_architecture = metadata.get("architecture")
        if checkpoint_architecture and checkpoint_architecture != settings["architecture"]:
            raise ValueError(
                f"Checkpoint architecture is {checkpoint_architecture!r}, expected "
                f"{settings['architecture']!r}."
            )
        model.to(self.device).eval()
        self.cache[display_name] = model
        self.metadata[display_name] = metadata
        return model

    def preload(self, display_name: str) -> str:
        started = time.perf_counter()
        self.load(display_name)
        elapsed = (time.perf_counter() - started) * 1000
        return f"Loaded {display_name} on {self.device} in {elapsed:.0f} ms"

    def predict(self, display_name: str, uploaded: list[Any] | None):
        if not uploaded:
            raise ValueError("Upload at least one de-identified tumor-bearing slice.")
        paths = sorted(
            (_uploaded_path(item) for item in uploaded),
            key=lambda path: natural_key(path.name),
        )
        paths = paths[: self.max_slices_per_patient]
        originals: list[Image.Image] = []
        tensors: list[torch.Tensor] = []
        for path in paths:
            with Image.open(path) as image:
                original = image.convert("RGB")
            originals.append(original)
            tensors.append(self.transform(original))
        bag = torch.stack(tensors).to(self.device)
        model = self.load(display_name)
        started = time.perf_counter()
        with torch.inference_mode():
            logits, attention, _ = forward_with_details(model, bag)
            probabilities = torch.softmax(logits, dim=-1).detach().cpu().tolist()
        elapsed = (time.perf_counter() - started) * 1000

        predicted = int(torch.tensor(probabilities).argmax().item())
        architecture = self.registry[display_name]["architecture"]
        gallery = []
        for index, (path, original, tensor, weight) in enumerate(
            zip(paths, originals, tensors, attention.detach().cpu().tolist(), strict=True)
        ):
            try:
                maps = smooth_gradcam_pp(
                    model,
                    tensor.unsqueeze(0).to(self.device),
                    backbone=architecture,
                    target_class=predicted,
                    mil_single_instance=True,
                )
                visual = overlay_heatmap(original, maps[0])
                note = ""
            except Exception as exc:
                visual = original
                note = f"; CAM unavailable ({type(exc).__name__})"
            caption = f"Slice {index + 1}; attention={weight:.4f}; file={path.name}{note}"
            gallery.append((visual, caption))

        result = {name: float(probabilities[index]) for index, name in enumerate(self.class_names)}
        metadata = self.metadata.get(display_name, {})
        details = (
            f"Model: {display_name}\n"
            f"Prediction: {self.class_names[predicted]}\n"
            f"Confidence: {probabilities[predicted]:.4f}\n"
            f"Processed slices: {len(paths)}\n"
            f"Inference time: {elapsed:.1f} ms\n"
            f"Device: {self.device}\n"
            f"Checkpoint epoch: {metadata.get('epoch', 'not recorded')}\n"
            "Research prototype only; not for clinical use."
        )
        return result, details, gallery


def _uploaded_path(item: Any) -> Path:
    value = getattr(item, "name", item)
    path = Path(str(value))
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.casefold() not in {".png", ".jpg", ".jpeg"}:
        raise ValueError(f"Unsupported image type: {path.suffix}")
    return path


def create_demo(registry_path: str | Path, device_name: str | None = None):
    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError("Install the 'app' extra to launch the Gradio prototype.") from exc

    inference = load_inference_settings(registry_path)
    service = ModelService(load_registry(registry_path), device_name=device_name, **inference)
    model_names = list(service.registry)
    css = """
    .gradio-container { font-family: 'Times New Roman', serif; }
    .gr-prose h1, .gr-prose h2, .gr-prose h3 { color: #2a6b9c; }
    """
    with gr.Blocks(title="PNI-MIL research prototype", theme=gr.themes.Soft(), css=css) as demo:
        gr.Markdown(
            "# PNI-MIL: CT image neural invasion research system\n"
            "Patient-level inference from ordered, de-identified tumor-bearing CECT slices. "
            "This interface is not a medical device."
        )
        with gr.Tabs():
            with gr.Tab("Patient inference"):
                with gr.Row():
                    with gr.Column():
                        selector = gr.Dropdown(model_names, value=model_names[0], label="MIL model")
                        files = gr.Files(
                            file_types=[".png", ".jpg", ".jpeg"], label="Patient slices"
                        )
                        preload = gr.Button("Preload model")
                        status = gr.Textbox(label="Model status")
                        run = gr.Button("Start analysis", variant="primary")
                    with gr.Column():
                        result = gr.Label(num_top_classes=2, label="Patient-level result")
                        details = gr.Textbox(lines=9, label="Run details")
                gallery = gr.Gallery(columns=4, label="Slice heatmap visualization")
            with gr.Tab("Use and limitations"):
                gr.Markdown(
                    "- Upload slices from one patient only and remove identifying information.\n"
                    f"- At most {service.max_slices_per_patient} naturally sorted slices are processed.\n"
                    "- Heatmaps are qualitative model-association displays.\n"
                    "- Do not use this research prototype for diagnosis or treatment decisions."
                )
        preload.click(service.preload, inputs=selector, outputs=status)
        run.click(service.predict, inputs=[selector, files], outputs=[result, details, gallery])
    return demo


def launch(
    registry_path: str | Path,
    *,
    device_name: str | None = None,
    share: bool = False,
) -> None:
    """Launch locally by default; public sharing must be requested explicitly."""
    demo = create_demo(registry_path, device_name=device_name)
    demo.launch(share=share)
