# External model weights

Model binaries are not stored in this GitHub repository. The four inference-only
PyTorch checkpoints are hosted in the public
[OSCC PNI MIL checkpoints](https://huggingface.co/changeindream/oscc-pni-mil-checkpoints)
repository on Hugging Face.

Download all four files and verify their published SHA-256 digests with:

```bash
python scripts/download_weights.py
```

`manifest.json` pins the Hugging Face revision, exact filenames, byte counts,
inference-file SHA-256 digests, and the SHA-256 digests of the original training
checkpoints from which the inference files were exported. Downloaded `.pt` files
are deliberately ignored by Git.

| Model | Hugging Face file |
| --- | --- |
| ResNet152-MIL | [`resnet_mil.pt`](https://huggingface.co/changeindream/oscc-pni-mil-checkpoints/blob/main/resnet_mil.pt) |
| DenseNet121-MIL | [`densenet_mil.pt`](https://huggingface.co/changeindream/oscc-pni-mil-checkpoints/blob/main/densenet_mil.pt) |
| Swin-Base-MIL | [`swin_mil.pt`](https://huggingface.co/changeindream/oscc-pni-mil-checkpoints/blob/main/swin_mil.pt) |
| ViT-Base-MIL | [`vit_mil.pt`](https://huggingface.co/changeindream/oscc-pni-mil-checkpoints/blob/main/vit_mil.pt) |

These models are research artifacts, not medical devices. Do not use their
output for diagnosis or treatment decisions. Licensing for the exact SHA-pinned
files is described in [`MODEL_WEIGHTS_LICENSE.md`](../MODEL_WEIGHTS_LICENSE.md);
upstream code, pretrained parameters, and datasets retain their own terms.
