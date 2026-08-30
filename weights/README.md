# Released inference weights

This directory contains the four patient-level MIL checkpoints used by the Gradio
research interface. The files contain model parameters and non-identifying
training metadata only; optimizer state was removed because it is not used for
inference. Removing optimizer state does not change model predictions.

The ResNet and DenseNet `.pt` files are tracked directly with Git LFS. The larger
Swin and ViT checkpoints are stored as 64 MiB Git LFS parts so every upload can
complete within GitHub's short-lived object-upload window. Reconstruct them after
cloning with:

```bash
python scripts/manage_weight_parts.py reconstruct
```

Reconstruction is lossless: the script verifies every part and the complete
checkpoint against the byte counts and SHA-256 digests in `manifest.json` before
atomically creating each `.pt` file. The resulting files are byte-for-byte
identical to the released inference checkpoints.

These models are research artifacts, not medical devices. Do not use their output
for diagnosis or treatment decisions.

## License

The project-authored checkpoint files in this directory are released under the
repository's [Apache License 2.0](../LICENSE) to the extent the authors own or
are authorized to license the relevant rights. Upstream code, pretrained
parameters, and datasets retain their own terms, as summarized in
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md). No patient data are
licensed or distributed with these checkpoints.
