# Local data preparation

Patient data are intentionally absent from this repository. Keep imaging data and all derived patient-level files outside the Git working tree whenever possible.

## De-identification requirements

- Remove DICOM headers and burned-in identifiers using an institution-approved workflow.
- Replace names and medical-record numbers with random study identifiers before running this code.
- Do not encode class labels, center names, or clinical outcomes in filenames.
- Maintain the re-identification key in an access-controlled clinical system, never beside the code.
- Inspect screenshots and exported overlays before publication; pixels can also contain identifiers.

## Recommended local layout

```text
private_data/
  nifti/
    images/<study_id>.nii.gz
    masks/<study_id>.nii.gz
  slices/
  manifest.csv
```

The `manifest.csv` schema is documented in the repository README. The code checks that each patient appears in only one split and flags identifiers that resemble names or medical-record numbers.

## Split policy

Create training, validation, and independent test assignments at the patient level, stratified by center and PNI status. Do not randomly split individual slices. The independent test set must not be used for fitting, threshold selection, or hyperparameter selection.
