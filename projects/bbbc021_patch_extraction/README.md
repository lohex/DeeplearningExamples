# BBBC021 patch extraction

## Goal

Build 200,000 balanced three-channel morphology patches from DMSO, cytochalasin B, nocodazole, taxol and AZ-A images for classification and Grad-CAM experiments.

## Contents

- `notebooks/build_patch_dataset.ipynb`: configuration, execution, storage estimate and validation
- `src/bbbc021_patch_extraction/pipeline.py`: metadata selection, group-aware splits, downloads, extraction, sharding and validation

## Run

```bash
python -m pip install -e ".[bbbc021]"
jupyter lab projects/bbbc021_patch_extraction/notebooks/build_patch_dataset.ipynb
```

Splits are assigned by plate and well before patch extraction. Patches from the same source field remain correlated, so downstream metrics should also be aggregated by field or well.
