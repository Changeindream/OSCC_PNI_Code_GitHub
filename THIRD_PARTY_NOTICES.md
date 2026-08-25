# Third-party notices

`src/oscc_pni/models/legacy_swin.py` is derived from the Microsoft Swin
Transformer reference implementation and the timm model implementation. The
source was retained because the released checkpoint uses its legacy stage
down-sampling layout and cannot be loaded strictly into recent timm Swin models.

- Microsoft Swin Transformer: MIT License, https://github.com/microsoft/Swin-Transformer
- timm: Apache License 2.0, https://github.com/huggingface/pytorch-image-models

PyTorch and torchvision model definitions are used through their public package
APIs and remain subject to their respective upstream licenses.
