# OSCC PNI prediction from contrast-enhanced CT

This repository contains the analysis code for the study **“Multicenter contrast-enhanced CT deep learning for predicting perineural invasion in oral squamous cell carcinoma.”** It implements the methods described in the manuscript and supplementary material while excluding patient data, exploratory tutorial code, and generated figures. The four de-identified, inference-only MIL checkpoints used by the Gradio interface are published as assets of the private `model-weights-v1` release.

The code covers:

- CT resampling, soft-tissue windowing, ROI masking, and tumor-bearing-slice export;
- handcrafted radiomics and frozen three-dimensional ResNet-18 feature extraction;
- exploratory clinical, radiomics, pretrained-feature, and direct-fusion classifiers;
- ROI-guided ResNet101, DenseNet121, ViT-Base, and Swin-Base training;
- attention-based patient-level multiple-instance learning (MIL);
- patient-level performance estimates, bootstrap confidence intervals, and publication-style plots;
- Grad-CAM-family maps, integrated gradients, occlusion, gradient SHAP, activation maximization, and UMAP;
- a local Gradio research prototype for patient-bag inference.

## Important scope statement

This is research software, not a medical device. It must not be used for clinical diagnosis or treatment decisions. The paper reports internal testing; prospective site-separated validation is still required.

No imaging data, masks, patient names, medical-record numbers, or derived patient-level outputs are included. The released checkpoints contain model tensors and primitive training metadata only. See [data/README.md](data/README.md) before preparing a local dataset.

## Released Gradio models

The deployment interface is tied to the exact architectures represented by the released weights:

| Interface model | Checkpoint-compatible architecture | Inference file |
| --- | --- | --- |
| ResNet152-MIL | torchvision ResNet-152 + normalized attention pooling | `weights/resnet152_mil_best_auc.pt` |
| DenseNet121-MIL | torchvision DenseNet-121 + normalized attention pooling | `weights/densenet121_mil_best_auc.pt` |
| Swin-Base-MIL | legacy Swin-Base + contextual MIL attention (256 dimensions) | `weights/swin_base_mil_best_auc.pt` |
| ViT-Base-MIL | timm ViT-B/16 + normalized attention pooling | `weights/vit_base_mil_best_auc.pt` |

The manuscript-aligned training pipeline and the released interface models are kept separate because the original Gradio checkpoint uses ResNet152, whereas the manuscript configuration reports ResNet101. The interface always loads released weights strictly and will fail if an architecture does not match.

## Installation

Python 3.11 is recommended. The released GPU environment uses PyTorch 2.7.1, torchvision 0.22.1, timm 1.0.15, and CUDA 12.8. Clone the repository, install the environment, and download the four private-release assets into `weights/` before launching the interface:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-app-cu128.txt
```

Authenticated GitHub CLI users can download all four checkpoints with:

```bash
gh release download model-weights-v1 \
  --repo Changeindream/OSCC_PNI_Code_GitHub \
  --dir weights
```

Alternatively, open the repository's **Releases** page and place the four `.pt`
assets in `weights/`. Verify their sizes and SHA-256 digests against
`weights/manifest.json` before inference.

The complete analysis extras can be installed with:

```bash
python -m pip install -e ".[radiomics,xai,app,baselines]"
```

## Data manifest

All two-dimensional and MIL commands use a CSV manifest rather than inferring labels from patient-identifying filenames. Required columns are:

| Column | Description |
| --- | --- |
| `patient_id` | De-identified, stable identifier |
| `image_path` | Path to one tumor-bearing slice |
| `label` | `0` for non-PNI and `1` for PNI |
| `split` | `train`, `validation`, or `test` |

Optional columns are `center`, `t_stage`, `mask_path`, `roi_area`, and `slice_index`. Every patient must occur in exactly one split. Run the leakage and privacy checks before training:

```bash
oscc-pni audit-manifest --manifest data/manifest.csv
```

## Paper-aligned configuration

The default parameters are stored in [configs/paper.yaml](configs/paper.yaml). This file is the single source of truth for the reported backbones, image size, optimizer settings, focal-loss parameters, MIL attention dimension, bootstrap count, and explainability parameters.

## Typical workflow

```bash
# 1. Convert paired NIfTI volumes and masks to tumor-bearing slices.
oscc-pni prepare-slices \
  --images-dir /path/to/images \
  --masks-dir /path/to/masks \
  --id-map /secure/path/to/id_map.csv \
  --label 1 \
  --split train \
  --output-dir /path/to/deidentified_slices \
  --manifest-out /path/to/manifest.csv \
  --apply-mask

# 2. Train one ROI-guided single-slice model.
oscc-pni train-slice \
  --manifest /path/to/manifest.csv \
  --backbone swin_base \
  --output-dir outputs/swin_slice

# 3. Initialize and train patient-level MIL.
oscc-pni train-mil \
  --manifest /path/to/manifest.csv \
  --backbone swin_base \
  --slice-checkpoint outputs/swin_slice/best.pt \
  --output-dir outputs/swin_mil

# 4. Evaluate a locked checkpoint on the independent test split.
oscc-pni evaluate-mil \
  --manifest /path/to/manifest.csv \
  --backbone swin_base \
  --checkpoint outputs/swin_mil/best.pt \
  --output-dir outputs/swin_mil_test

# 5. Launch the local research interface with the released weights.
oscc-pni app --registry configs/model_registry.yaml
```

Paths in training examples are placeholders. Public sharing is disabled by default. The optional `--share` flag must never be used with identifiable images.

## Reproducibility notes

- Patient-level splitting is mandatory. Slice-level random splitting is rejected by the manifest audit.
- Training bags may be randomly subsampled; validation and test bags always use all tumor-bearing slices.
- The decision threshold is selected in the validation set using Youden's index, saved with the checkpoint, and locked for independent testing.
- Test-set results must not be used for model selection or further fitting.
- Stochastic GPU kernels and data-loader behavior can still produce small run-to-run differences. Seeds and deterministic settings are recorded in each run directory.
- ImageNet pretrained weights are downloaded by `torchvision`/`timm` for new training unless a local cache is already present. Released inference checkpoints do not download additional backbone weights.

## Repository map

```text
configs/                 Paper-aligned configuration and local registry template
data/                    Data-format and privacy guidance only
docs/                    Code-selection audit and reproducibility notes
scripts/                 Trusted checkpoint export and verification utilities
src/oscc_pni/            Reusable implementation
tests/                   Fast unit tests that do not require patient data
weights/                 Checkpoint manifest and downloaded release assets
```

See [docs/CODE_SELECTION.md](docs/CODE_SELECTION.md) for the manuscript-to-code mapping and the reasons original files were retained, consolidated, or excluded.

## Citation and license

.
