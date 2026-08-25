from __future__ import annotations

import gc
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "pni-matplotlib"))

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError
from torchvision import transforms

try:
    import spaces
except ImportError:  # Local and conventional GPU/CPU deployments.
    class _SpacesFallback:
        @staticmethod
        def GPU(function=None, **_kwargs):
            def decorator(func):
                return func

            return decorator(function) if function is not None else decorator

    spaces = _SpacesFallback()


APP_ROOT = Path(__file__).resolve().parent
MODEL_REPO_ID = os.environ.get("MODEL_REPO_ID", "changeindream/oscc-pni-mil-checkpoints")
CLASS_NAMES = ("Non-PNI", "PNI")
MAX_SLICES = 20
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


def _discover_example_bags() -> tuple[list[list[list[str]]], list[str]]:
    """Return anonymous patient bags in the shape expected by one gr.Files input."""
    roots = (APP_ROOT.parent / "examples", APP_ROOT / "examples")
    example_root = next((path for path in roots if path.is_dir()), None)
    if example_root is None:
        return [], []

    rows: list[list[list[str]]] = []
    labels: list[str] = []
    for directory_name, display_name in (("pni", "PNI"), ("non_pni", "Non-PNI")):
        class_directory = example_root / directory_name
        for case_directory in sorted(class_directory.glob("*")):
            if not case_directory.is_dir():
                continue
            slices = sorted(case_directory.glob("slice_*.png"))
            if not slices:
                continue
            rows.append([[str(path.resolve()) for path in slices]])
            labels.append(f"{display_name}: {case_directory.name} ({len(slices)} slices)")
    return rows, labels


DEMO_EXAMPLES, DEMO_EXAMPLE_LABELS = _discover_example_bags()

MODEL_SPECS = {
    "Swin Transformer MIL": {
        "kind": "swin",
        "filename": "swin_mil.pt",
        "weight": APP_ROOT / "weights" / "swin_mil.pt",
        "architecture": "Swin-Base + gated multi-head MIL attention",
    },
    "Vision Transformer MIL": {
        "kind": "vit",
        "filename": "vit_mil.pt",
        "weight": APP_ROOT / "weights" / "vit_mil.pt",
        "architecture": "ViT-Base/16 + attention MIL",
    },
    "ResNet-152 MIL": {
        "kind": "resnet",
        "filename": "resnet152_mil.pt",
        "weight": APP_ROOT / "weights" / "resnet152_mil.pt",
        "architecture": "ResNet-152 + normalized attention MIL",
    },
    "DenseNet MIL": {
        "kind": "densenet",
        "filename": "densenet121_mil.pt",
        "weight": APP_ROOT / "weights" / "densenet121_mil.pt",
        "architecture": "DenseNet-121 + attention MIL",
    },
}

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ]
)

_MODEL_LOCK = threading.RLock()
_MODEL_CACHE: dict[str, Any] = {"name": None, "model": None, "device": None}


def _runtime_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _release_cached_model() -> None:
    model = _MODEL_CACHE.get("model")
    if model is not None:
        try:
            model.to("cpu")
        except Exception:
            pass
    _MODEL_CACHE.update(name=None, model=None, device=None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_model(kind: str) -> nn.Module:
    if kind == "resnet":
        from model_defs.resnet152_mil import MILModel

        return MILModel(num_classes=2, use_pretrained=False)
    if kind == "densenet":
        from model_defs.densenet121_mil import DenseNet121MILModel

        return DenseNet121MILModel(num_classes=2, use_pretrained=False)
    if kind == "swin":
        from model_defs.swin_mil import SwinMILModel

        return SwinMILModel(
            model_name="swin_base_patch4_window7_224",
            num_classes=2,
            attention_dim=256,
            pretrained=False,
        )
    if kind == "vit":
        from model_defs.vit_mil import ViTMILModel

        return ViTMILModel(num_classes=2, pretrained=False)
    raise ValueError(f"Unsupported model kind: {kind}")


def _resolve_weight_path(spec: dict[str, Any]) -> Path:
    local_path = Path(spec["weight"])
    if local_path.is_file():
        return local_path

    try:
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                repo_id=MODEL_REPO_ID,
                filename=spec["filename"],
                repo_type="model",
            )
        )
    except Exception as exc:
        raise FileNotFoundError(
            f"Inference weight is unavailable locally and could not be downloaded: {spec['filename']} "
            f"from {MODEL_REPO_ID}."
        ) from exc


def _read_state_dict(weight_path: Path):
    try:
        return torch.load(
            weight_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        return torch.load(weight_path, map_location="cpu")


def load_model(model_name: str) -> tuple[nn.Module, torch.device]:
    if model_name not in MODEL_SPECS:
        raise ValueError("The selected model is not available.")

    with _MODEL_LOCK:
        current = _MODEL_CACHE
        if current["name"] == model_name and current["model"] is not None:
            return current["model"], current["device"]

        _release_cached_model()
        spec = MODEL_SPECS[model_name]
        model = _build_model(spec["kind"])
        state_dict = _read_state_dict(_resolve_weight_path(spec))
        model.load_state_dict(state_dict, strict=True)
        device = _runtime_device()
        model = model.to(device).eval()
        _MODEL_CACHE.update(name=model_name, model=model, device=device)
        return model, device


def _file_path(item: Any) -> Path:
    value = item.name if hasattr(item, "name") else item
    return Path(str(value))


def _slice_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)(?=\.[^.]+$)", path.name)
    return (int(match.group(1)) if match else 10**12, path.name.casefold())


def _validated_images(files: list[Any] | None) -> tuple[list[Path], list[Image.Image]]:
    if not files:
        raise ValueError("Upload at least one prepared tumor-bearing slice.")
    if len(files) > MAX_SLICES:
        raise ValueError(f"A maximum of {MAX_SLICES} slices is accepted per bag.")

    paths = sorted((_file_path(item) for item in files), key=_slice_sort_key)
    images: list[Image.Image] = []
    for path in paths:
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            raise ValueError("Only PNG and JPEG images are supported; DICOM is not accepted.")
        if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError("Each image must be a readable file no larger than 20 MB.")
        try:
            with Image.open(path) as probe:
                probe.verify()
            with Image.open(path) as source:
                source = ImageOps.exif_transpose(source)
                width, height = source.size
                if width * height > MAX_IMAGE_PIXELS:
                    raise ValueError("An uploaded image exceeds the 20-megapixel safety limit.")
                images.append(source.convert("RGB").copy())
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("One or more uploads are not valid PNG/JPEG images.") from exc
    return paths, images


def _forward_with_attention(
    model_name: str,
    model: nn.Module,
    bag: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    kind = MODEL_SPECS[model_name]["kind"]
    if kind in {"resnet", "densenet"}:
        features = model.feature_extractor(bag)
        if features.ndim > 2:
            features = features.reshape(features.shape[0], -1)
        attention = F.softmax(model.attention_net(features).squeeze(-1), dim=0)
        bag_feature = torch.sum(features * attention.unsqueeze(-1), dim=0)
        logits = model.classifier(bag_feature)
        return logits, attention
    if kind == "swin":
        logits, attention = model(bag, return_attention=True)
        return logits, attention.reshape(-1)
    if kind == "vit":
        logits, attention, _features = model(bag)
        return logits, attention.reshape(-1)
    raise ValueError("Unsupported model configuration.")


def _font(size: int = 18):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _attention_card(image: Image.Image, rank: int, weight: float, peak: float) -> Image.Image:
    canvas = ImageOps.contain(image, (420, 420), Image.Resampling.LANCZOS).copy()
    intensity = 0.0 if peak <= 0 else min(1.0, weight / peak)
    border = (28, int(125 + 90 * intensity), int(150 - 55 * intensity))
    canvas = ImageOps.expand(canvas, border=5, fill=border)
    draw = ImageDraw.Draw(canvas, "RGBA")
    label = f"Slice {rank:02d}  |  MIL attention {weight:.4f}"
    draw.rounded_rectangle((8, 8, min(canvas.width - 8, 330), 43), radius=8, fill=(8, 25, 44, 205))
    draw.text((18, 15), label, fill=(255, 255, 255, 255), font=_font(16))
    return canvas


def _reshape_tokens(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 3:
        return tensor
    batch, tokens, channels = tensor.shape
    if tokens > 1 and int((tokens - 1) ** 0.5) ** 2 == tokens - 1:
        tensor = tensor[:, 1:, :]
        tokens -= 1
    side = int(tokens**0.5)
    if side * side != tokens:
        return tensor
    return tensor.reshape(batch, side, side, channels).permute(0, 3, 1, 2)


def _target_layer(model_name: str, model: nn.Module):
    kind = MODEL_SPECS[model_name]["kind"]
    if kind == "resnet":
        return model.feature_extractor.layer4[-1], None
    if kind == "densenet":
        return model.feature_extractor.features.denseblock4, None
    if kind == "vit":
        return model.vit_backbone.blocks[-1].norm1, _reshape_tokens
    if kind == "swin":
        return model.backbone.layers[-1].blocks[-1].norm1, _reshape_tokens
    return None, None


class _CamWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        output = self.model(tensor)
        if isinstance(output, tuple):
            output = output[0]
        return output.unsqueeze(0) if output.ndim == 1 else output


def _cam_overlays(
    model_name: str,
    model: nn.Module,
    bag: torch.Tensor,
    raw_images: list[Image.Image],
    attention: np.ndarray,
    target_class: int,
) -> list[tuple[Image.Image, str]]:
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    layer, reshape = _target_layer(model_name, model)
    if layer is None:
        return []
    output: list[tuple[Image.Image, str]] = []
    with GradCAMPlusPlus(
        model=_CamWrapper(model),
        target_layers=[layer],
        reshape_transform=reshape,
    ) as cam:
        for index in range(len(raw_images)):
            cam_map = cam(
                input_tensor=bag[index : index + 1],
                targets=[ClassifierOutputTarget(target_class)],
            )[0]
            base = np.asarray(raw_images[index].resize((224, 224))).astype(np.float32)
            color = cv2.cvtColor(
                cv2.applyColorMap(np.uint8(np.clip(cam_map, 0, 1) * 255), cv2.COLORMAP_TURBO),
                cv2.COLOR_BGR2RGB,
            ).astype(np.float32)
            overlay = Image.fromarray(np.uint8(np.clip(0.58 * base + 0.42 * color, 0, 255)))
            caption = f"Slice {index + 1:02d} | Grad-CAM++ | attention {attention[index]:.4f}"
            output.append((overlay, caption))
    return output


@spaces.GPU(duration=120)
def preload_selected_model(model_name: str) -> str:
    started = time.perf_counter()
    try:
        _model, device = load_model(model_name)
        elapsed = time.perf_counter() - started
        status = (
            f"Model Loaded: {model_name} | Device: {device} | "
            f"Time: {elapsed * 1000:.1f}ms"
        )
        if "checkpoint_note" in MODEL_SPECS[model_name]:
            status += f"\nNotice: {MODEL_SPECS[model_name]['checkpoint_note']}"
        return status
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return (
            f"Load Failed: {model_name} | Error: {type(exc).__name__}: {exc} | "
            f"Time: {elapsed * 1000:.1f}ms"
        )


def _predict_patient_outputs(
    model_name: str,
    files: list[Any] | None,
    make_cam: bool,
    deidentified_confirmation: bool,
):
    if not deidentified_confirmation:
        raise gr.Error("Confirm that all uploaded images are de-identified and permitted for research use.")

    _paths, raw_images = _validated_images(files)
    model, device = load_model(model_name)
    bag = torch.stack([TRANSFORM(image) for image in raw_images]).to(device)

    started = time.perf_counter()
    with torch.inference_mode():
        logits, attention = _forward_with_attention(model_name, model, bag)
        probabilities = F.softmax(logits.reshape(-1), dim=-1)
    elapsed = time.perf_counter() - started

    probs = probabilities.detach().float().cpu().numpy()
    weights = attention.detach().float().cpu().numpy()
    if probs.shape != (2,) or weights.shape != (len(raw_images),):
        raise RuntimeError("Unexpected model output shape; inference was stopped.")

    predicted = int(np.argmax(probs))
    peak = float(weights.max(initial=0.0))
    gallery = [
        (_attention_card(image, index + 1, float(weights[index]), peak), f"Slice {index + 1:02d}")
        for index, image in enumerate(raw_images)
    ]

    cam_note = "Not requested"
    if make_cam:
        try:
            cam_gallery = _cam_overlays(
                model_name,
                model,
                bag,
                raw_images,
                weights,
                predicted,
            )
            if len(cam_gallery) != len(raw_images):
                raise RuntimeError("Grad-CAM++ did not return one overlay per uploaded slice.")
            gallery = cam_gallery
            cam_note = "Grad-CAM++ generated for all uploaded slices"
        except Exception as exc:
            cam_note = f"Grad-CAM++ unavailable ({type(exc).__name__}); MIL attention cards shown"

    result = {
        CLASS_NAMES[0]: float(probs[0]),
        CLASS_NAMES[1]: float(probs[1]),
    }
    summary = (
        "### Research model output\n"
        f"**Predicted class:** {CLASS_NAMES[predicted]}  \n"
        f"**PNI probability:** {probs[1]:.4f}  \n"
        f"**Model:** {model_name}  \n"
        f"**Slices processed:** {len(raw_images)} of {MAX_SLICES} maximum  \n"
        f"**Forward-pass time:** {elapsed:.2f} s on {device.type.upper()}  \n"
        f"**Visualization:** {cam_note}\n\n"
        "> This output is experimental and is not a diagnosis or treatment recommendation."
    )
    details = {
        "model": model_name,
        "architecture": MODEL_SPECS[model_name]["architecture"],
        "runtime": device.type.upper(),
        "slices_processed": len(raw_images),
        "probabilities": result,
        "attention_weights_in_upload_order": [round(float(value), 6) for value in weights],
        "forward_pass_seconds": round(elapsed, 4),
        "clinical_use": False,
    }
    if "checkpoint_note" in MODEL_SPECS[model_name]:
        details["checkpoint_note"] = MODEL_SPECS[model_name]["checkpoint_note"]
    return result, summary, gallery, details


@spaces.GPU(duration=180)
def predict_patient(model_name: str, files: list[Any] | None):
    result, _summary, gallery, details = _predict_patient_outputs(
        model_name,
        files,
        make_cam=True,
        deidentified_confirmation=True,
    )
    predicted = max(result, key=result.get)
    confidence = result[predicted]
    info_lines = [
        "Patient ID: not displayed for privacy",
        f"Model: {model_name}",
        f"Prediction: {predicted}",
        f"Confidence: {confidence:.4f}",
        f"Slices Processed: {details['slices_processed']}",
        f"Inference Time: {details['forward_pass_seconds'] * 1000:.1f}ms",
        f"Device: {details['runtime']}",
        "Research use only; this output is not a clinical diagnosis.",
    ]
    if "checkpoint_note" in details:
        info_lines.append(f"Notice: {details['checkpoint_note']}")
    return result, "\n".join(info_lines), gallery


CSS = """
.gradio-container {
    font-family: 'Times New Roman', serif;
}
.gr-prose h1, h2, h3 {
    color: #2a6b9c;
    font-family: 'Times New Roman', serif;
}
.prediction-info {
    background-color: #f7f7f7;
    border-radius: 8px;
    padding: 10px;
    margin-top: 10px;
    border-left: 4px solid #2a6b9c;
    font-family: 'Times New Roman', serif;
}
.footer {
    margin-top: 20px;
    text-align: center;
    font-size: 0.8em;
    color: #666;
    font-family: 'Times New Roman', serif;
}
"""

ENGLISH_HEAD = r"""
<script>
(() => {
    document.documentElement.lang = "en";

    for (const [property, value] of [
        ["language", "en-US"],
        ["languages", ["en-US", "en"]],
    ]) {
        try {
            Object.defineProperty(window.navigator, property, {
                configurable: true,
                get: () => value,
            });
        } catch (_) {
            // The visible-text fallback below still guarantees an English UI.
        }
    }

    const translations = new Map([
        ["将文件拖放到此处 - 或 - 点击上传", "Drop files here - or - click to upload"],
        ["将文件拖放到此处", "Drop files here"],
        ["点击上传", "Click to upload"],
        ["- 或 -", "- or -"],
        ["使用 Gradio 构建", "Built with Gradio"],
        ["加载中...", "Loading..."],
        ["正在加载...", "Loading..."],
        ["清除", "Clear"],
        ["下载", "Download"],
        ["上传", "Upload"],
        ["删除", "Delete"],
        ["编辑", "Edit"],
        ["全屏", "Fullscreen"],
        ["关闭", "Close"],
        ["上一个", "Previous"],
        ["下一个", "Next"],
        ["复制", "Copy"],
        ["提交", "Submit"],
        ["取消", "Cancel"],
        ["标志", "logo"],
    ]);

    const translateString = (value) => {
        let translated = value;
        for (const [source, target] of translations) {
            translated = translated.split(source).join(target);
        }
        return translated;
    };

    const translateNode = (node) => {
        if (node.nodeType === Node.TEXT_NODE) {
            const translated = translateString(node.nodeValue || "");
            if (translated !== node.nodeValue) node.nodeValue = translated;
            return;
        }
        if (node.nodeType !== Node.ELEMENT_NODE) return;

        for (const attribute of ["aria-label", "title", "placeholder", "alt"]) {
            const current = node.getAttribute(attribute);
            if (current === null) continue;
            const translated = translateString(current);
            if (translated !== current) node.setAttribute(attribute, translated);
        }
        for (const child of node.childNodes) translateNode(child);
    };

    const startEnglishUI = () => {
        translateNode(document.documentElement);
        new MutationObserver((mutations) => {
            for (const mutation of mutations) {
                if (mutation.type === "characterData") translateNode(mutation.target);
                for (const node of mutation.addedNodes || []) translateNode(node);
                if (mutation.type === "attributes") translateNode(mutation.target);
            }
        }).observe(document.documentElement, {
            childList: true,
            subtree: true,
            characterData: true,
            attributes: true,
            attributeFilter: ["aria-label", "title", "placeholder", "alt"],
        });
    };

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", startEnglishUI, {once: true});
    } else {
        startEnglishUI();
    }
})();
</script>
"""

MODEL_CHOICES = [
    "ResNet-152 MIL",
    "DenseNet MIL",
    "Swin Transformer MIL",
    "Vision Transformer MIL",
]

with gr.Blocks(
    css=CSS,
    head=ENGLISH_HEAD,
    theme=gr.themes.Soft(primary_hue="blue"),
    analytics_enabled=False,
    delete_cache=(3600, 3600),
    title="PNI-MIL: CT Image Neural Invasion Diagnosis System",
) as demo:
    gr.Markdown("# 🏥 PNI-MIL: CT Image Neural Invasion Diagnosis System")

    with gr.Tabs():
        with gr.TabItem("📊 Patient Inference"):
            with gr.Row():
                with gr.Column(scale=1):
                    model_select = gr.Dropdown(
                        choices=MODEL_CHOICES,
                        value=MODEL_CHOICES[0],
                        label="Select MIL Model",
                    )
                    input_files = gr.Files(
                        file_types=[".png", ".jpg", ".jpeg"],
                        file_count="multiple",
                        type="filepath",
                        label="Upload Patient Slices",
                    )
                    if DEMO_EXAMPLES:
                        gr.Examples(
                            examples=DEMO_EXAMPLES,
                            inputs=[input_files],
                            example_labels=DEMO_EXAMPLE_LABELS,
                            examples_per_page=10,
                            cache_examples=False,
                            api_name=False,
                            label="De-identified Example Patients",
                        )
                    preload_button = gr.Button("Preload Model")
                    preload_status = gr.Textbox(label="Model Status", lines=2)
                    run_button = gr.Button("Start Analysis", variant="primary")
                with gr.Column(scale=1):
                    label_output = gr.Label(
                        num_top_classes=len(CLASS_NAMES),
                        label="Diagnosis Result",
                    )
                    info_output = gr.Textbox(
                        label="Analysis Details",
                        lines=6,
                        elem_classes="prediction-info",
                    )

            gallery = gr.Gallery(
                label="Slice Heatmap Visualization",
                columns=4,
                height="auto",
            )

        with gr.TabItem("ℹ️ User Guide"):
            gr.Markdown(
                f"""
                ### Upload Instructions
                - Please upload sequential slices for the same patient (PNG/JPG/JPEG).
                - Filenames may contain underscore-separated slice numbers, e.g., `PATIENT_001.png`.
                - The system automatically sorts numeric filename suffixes and produces a patient-level MIL prediction.
                - Upload no more than {MAX_SLICES} de-identified, prepared tumor-bearing slices from one patient.
                - Click one of the bundled example patients to load all slices from that patient. Do not mix cases.

                ### Research and privacy notice
                - This internally evaluated research prototype is not a medical device and must not be used for diagnosis,
                  triage, prognosis, treatment decisions, or other clinical care.
                - Upload only images you are authorized to process. Do not upload DICOM files, names, medical record
                  numbers, dates of birth, burned-in annotations, or other identifiable information.
                - Probabilities and heatmaps are experimental model outputs, not calibrated clinical risks or causal
                  explanations.

                """
            )

    run_button.click(
        predict_patient,
        inputs=[model_select, input_files],
        outputs=[label_output, info_output, gallery],
        api_name=False,
        concurrency_limit=1,
    )
    preload_button.click(
        preload_selected_model,
        inputs=model_select,
        outputs=preload_status,
        api_name=False,
        concurrency_limit=1,
    )

    gr.Markdown(
        """<div class="footer">© 2025 PNI-MIL Diagnostic System | Deep Learning-based Medical Image Analysis Platform</div>"""
    )

demo.queue(default_concurrency_limit=1, max_size=8)

if __name__ == "__main__":
    demo.launch(show_api=False)

