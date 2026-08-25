---
title: OSCC PNI MIL Research Demo
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
python_version: 3.10
app_file: app.py
pinned: false
suggested_hardware: t4-small
license: other
---

# OSCC PNI patient-level MIL research demo

This directory contains the deployable source for the live
[Hugging Face Space](https://huggingface.co/spaces/changeindream/oscc-pni-mil-research-demo).
It provides patient-level multiple-instance-learning inference on prepared contrast-enhanced CT tumor slices and
supports the four released checkpoints: ResNet-152 MIL, DenseNet-121 MIL, Swin-Base MIL, and ViT-Base MIL.

When this application is launched from the repository, the ten anonymous patient bags in `../examples/` appear as
clickable examples. Each row loads all slices from one patient; slices from different rows must not be mixed.

The source was imported from Space commit `a057ed9` on 2026-08-26. The GitHub copy adds the bundled anonymous-example
selector and corrects the ResNet display label to match its ResNet-152 checkpoint; model inference behavior is unchanged.

## Research-use and privacy notice

This prototype is not a medical device and is not intended for diagnosis, triage, prognosis, or treatment decisions.
The underlying models have not completed prospective, site-separated external validation. Upload only de-identified
PNG/JPEG images that you are authorized to process. Do not upload DICOM files, protected health information, or images
with burned-in identifiers.

## Deployment notes

- ZeroGPU or a conventional NVIDIA GPU is recommended. CPU execution is supported but can be slow, especially for
  transformer models and Grad-CAM++.
- Only one model is held in memory at a time. Switching models releases the previous one.
- If a local `weights/` file is absent, the app downloads it on demand from the public
  `changeindream/oscc-pni-mil-checkpoints` model repository. Set `MODEL_REPO_ID` to override that source.
- Training optimizer states were removed from the uploaded files. `weights/manifest.json` records SHA-256 checksums of
  both the original checkpoints and the exact inference-only exports.
- Image preprocessing matches the supplied prototype: resize to 224 × 224, convert to RGB tensor, and apply ImageNet
  normalization.
- Uploaded files are transient inputs. Gradio cache cleanup is configured, but the uploader remains responsible for
  de-identification and permission to process the data.

## Local launch

```bash
python -m pip install -r requirements.txt
python app.py
```

From the repository root, use `python demo/app.py` after installing `demo/requirements.txt`.

The permanent public URL is supplied by the Hugging Face Space. A `gradio.live` link is intentionally not used because
it is a temporary tunnel to a local computer.
