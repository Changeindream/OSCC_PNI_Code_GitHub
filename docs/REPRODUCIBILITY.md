# Reproducibility and reporting checklist

Before releasing results from this repository:

1. Run `audit-manifest` and confirm zero patient overlap across splits.
2. Record a cryptographic hash of the final manifest without publishing the manifest itself.
3. Save the resolved YAML configuration, package versions, seed, device, and checkpoint hash with every run.
4. Select the classification threshold using validation predictions only.
5. Evaluate the locked checkpoint once on the independent test split.
6. Bootstrap at the patient level, not the slice level.
7. Describe slice-level DeLong tests, calibration, and decision curves as exploratory when within-patient dependence is not modeled.
8. Treat attribution maps as model-association displays, not evidence of biological mechanisms.
9. Verify that all exported images, paths, logs, and metadata are de-identified.
10. Release only inference-only weights that have been checked for non-tensor metadata, documented with SHA-256, and approved under the applicable institutional and ethics requirements.

## Expected cohort accounting

The submitted study reports 568 patients divided into training (`n=397`), validation (`n=113`), and independent test (`n=58`) sets. The corresponding slice counts are 7,343, 2,098, and 1,049. The code does not hard-code these values; an author-side run should compare the audited manifest summary with these reported counts and document any discrepancy.
