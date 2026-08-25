# Released inference weights

This directory contains the four patient-level MIL checkpoints used by the Gradio
research interface. The files contain model parameters and non-identifying
training metadata only; optimizer state was removed because it is not used for
inference. Removing optimizer state does not change model predictions.

The `.pt` files are distributed as assets of the private `model-weights-v1`
GitHub release so that the source repository stays lightweight. Download the four
assets into this directory before inference. File sizes and SHA-256 digests are
recorded in `manifest.json` and verified by the test suite.

Authenticated GitHub CLI users can run:

```bash
gh release download model-weights-v1 \
  --repo Changeindream/OSCC_PNI_Code_GitHub \
  --dir weights
```

These models are research artifacts, not medical devices. Do not use their output
for diagnosis or treatment decisions.
