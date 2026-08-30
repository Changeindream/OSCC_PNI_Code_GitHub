# Code-selection and refactoring record

This file records how the original working directories were translated into the public repository. The manuscript and supplementary material are the methodological authority; original filenames alone were not treated as evidence that a script contributed to the reported analysis.

## Retained and consolidated functionality

| Manuscript component | Original code area | Public implementation |
| --- | --- | --- |
| CT windowing, ROI masking, tumor-bearing slices | `深度学习代码/tools/切取roi.py`, `切取最大roi.py` | `oscc_pni.data.preprocessing` |
| Patient-level splitting and leakage checks | `数据集分割_按患者.py`, `检查标签泄露.py` | `oscc_pni.data.manifest` and `oscc_pni.privacy` |
| PyRadiomics feature extraction | `影像组学代码/1--提取影像组学特征.py` | `oscc_pni.features.radiomics` |
| Frozen 3D ResNet-18 features | `影像组学代码/2--提取深度学习特征.py` | `oscc_pni.features.deep_features` |
| Feature selection, classifiers, direct fusion | `3.3`, `5`, `7.1`, `8`, `8.1`, `9` scripts | `oscc_pni.features.classical` |
| Four ROI-guided end-to-end models | ResNet, DenseNet, ViT, and Swin training directories | `oscc_pni.models.backbones`, `oscc_pni.training.engine` |
| Attention-based patient-level MIL | Four duplicated MIL directories | `oscc_pni.models.mil`, `oscc_pni.training.engine` |
| Slice- and patient-level evaluation | `model comparison` directory | `oscc_pni.evaluation.metrics`, `oscc_pni.evaluation.plots` |
| Pixel attribution and representation analysis | `可解释性分析`, `visualization_toolkit`, `transformer_visualization_toolkit` | `oscc_pni.explainability` |
| Local research interface | `gradio_mil_patient_inference.py` | `oscc_pni.app` |

## Excluded material

- Patient CT slices, masks, filenames containing names or medical-record numbers, and patient-level JSON/CSV outputs.
- Redundant pretrained files and optimizer-heavy training checkpoints. The four Gradio MIL checkpoints are exported as inference-only `.pt` files, hosted in the public Hugging Face model repository, and verified against the SHA-256 digests pinned in `weights/manifest.json`; no model binaries are stored in GitHub.
- Generated JPG, PNG, PDF, and SVG figures; these are outputs, not source code.
- `__pycache__`, `.pyc`, debug logs, timestamped result folders, and repeated copies of identical modules.
- ImageNet, fruit, wine, and MNIST tutorial notebooks unrelated to the OSCC analysis.
- One-off color demonstrations, WPS compatibility tests, grid-search experiments, continuation scripts, and diagnostic helpers that were not described in the paper.
- Hard-coded Windows paths and local `share=True` Gradio deployment settings.

## Resolved manuscript/code drift

The original directory contains multiple chronological experiments. Several active-looking scripts conflict with the submitted methods:

- ResNet scripts frequently instantiate ResNet152, while the manuscript reports ResNet101.
- One ViT training script uses ViT-Small, while the manuscript reports ViT-Base.
- A DenseNet script uses batch size 16, while Supplementary Table S3 reports 32.
- One Swin-MIL module adds multi-head gated aggregation not described in the manuscript; the public code uses the reported normalized attention pooling.
- Older preprocessing scripts use a `-100 to 400 HU` window and 128-pixel output, whereas the submitted methods specify `-135 to 215 HU`, 1-mm isotropic resampling, and 224-pixel network input.
- Original split utilities expose patient names and medical-record numbers through filenames. The public code requires de-identified manifest IDs.

The manuscript-aligned training defaults continue to follow the manuscript and Supplementary Tables S3-S5. The Gradio deployment path is now explicitly separate and uses checkpoint-compatible ResNet152, DenseNet121, Swin-Base, and ViT-Base definitions. Every released checkpoint is loaded strictly, so the interface cannot silently combine a weight file with the wrong architecture.
