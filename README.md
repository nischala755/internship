# DRDO Internship: Polarimetric ISAR Recognition and Detection

This repository contains the notebooks, supporting Python modules, experiment
utilities, reports and presentations developed during the DRDO internship on
polarimetric ISAR space-target recognition and component-level oriented object
detection.

## Repository contents

- `notebooks/canonical/`: clean project and Colab-ready notebook versions.
- `notebooks/history/`: historical/result-bearing notebook variants retained
  for experiment traceability.
- `notebooks/guide_versions/`: notebook copies prepared for guide review.
- `notebooks/sources/`: percent-format Python sources used to build notebooks.
- `src/`: reusable feature extraction, normalization, feature selection and
  paper-aligned partitioning modules.
- `tools/`: notebook builders, plotting utilities and report generators.
- `tests/`: structural and helper-module tests.
- `docs/`: technical progress notes, paper summary and internship diary.
- `presentations/`: project review presentations.

## Main work streams

1. Full-polarimetric HH/HV/VH/VV preprocessing and visualization.
2. Conventional ResNet18 and complex-valued CNN classification.
3. Train-only normalization, feature selection and leakage analysis.
4. RAM-safe TFPSSTD processing and classification.
5. Component detection with DOTA/YOLO oriented bounding boxes.
6. Single-VV, Pauli and learnable polarimetric fusion comparisons.
7. Non-YOLO CenterNet/DETR diagnostic baselines and MMRotate planning.
8. Paper-aligned partitioning, long-epoch models and grouped K-fold validation.
9. Four-channel `imgaug` augmentation with resume-safe 100-epoch training.

## Data

The Pol-ISARSpaceTarget-1.0 and TFPSSTD datasets are not committed because of
their size and distribution terms. The notebooks expect data under Google
Drive paths documented inside each notebook, commonly:

```text
/content/drive/MyDrive/drdo/Pol_ISARSpaceTarget1.0
/content/tfpsstd
```

Credentials are never stored in the repository. IEEE DataPort notebooks prompt
for AWS credentials at runtime using `getpass` or environment variables.

## Reproducibility notes

- Treat historical notebooks as experiment records; use the canonical or
  explicitly `FIXED` notebook for a new run.
- Check each notebook's split protocol before comparing metrics.
- Normalization statistics are intended to be fitted on training data only.
- Long-running Colab experiments save checkpoints to Google Drive and support
  resume after disconnection.

