# De-identified demonstration cases

This directory contains ten randomly selected, prepared tumor-ROI patient bags from the held-out test collection:
five PNI cases and five non-PNI cases. The 80 PNG slices are included only to demonstrate the research interface.

Privacy safeguards applied before Git inclusion:

- original names and record numbers were removed from every path;
- cases were assigned new `PNI_###` or `NPNI_###` identifiers;
- source-to-alias mappings and the random-selection seed were not retained;
- every image was decoded and re-encoded as a new PNG without embedded metadata;
- full-resolution contact sheets were manually reviewed for burned-in text;
- file paths, metadata, dimensions, and SHA-256 checksums are verified against `manifest.json`.

To try one patient, select every `slice_*.png` file inside a single case directory. Never combine slices from different
case directories. Ground-truth class names are visible because these cases are interface examples; predictions remain
experimental and are not clinical results.

Before making this repository public, the repository owner remains responsible for confirming that institutional
approvals and participant permissions allow redistribution of these derived images.

Run the privacy and integrity check with:

```bash
python scripts/verify_demo_examples.py
```
