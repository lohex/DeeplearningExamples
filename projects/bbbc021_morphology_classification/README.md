# BBBC021 morphology classification

## Goal

Build a reproducible, leakage-aware microscopy patch dataset for five interpretable BBBC021 treatment classes: DMSO, cytochalasin B, nocodazole, taxol and AZ-A. The current project focuses on the data foundation required before a trustworthy classifier or Grad-CAM analysis can be trained.

## ML engineering questions

- What is the independent biological unit: patch, field, well or plate?
- How can splits be assigned before patch extraction so correlated crops cannot cross fold boundaries?
- How can class balance be controlled without discarding provenance?
- How much storage will a large patch dataset require, and how can a small pilot validate that estimate?
- How can downloading, preprocessing and sharding be resumed and audited?
- At which biological level should a later classifier be evaluated?

## Notebook workflow

[`01_build_patch_dataset.ipynb`](notebooks/01_build_patch_dataset.ipynb) selects treatments and source wells, assigns group-aware folds, downloads only required plates, estimates storage with a pilot and generates balanced WebDataset-compatible tar shards plus a manifest.

[Open notebook 01 in Colab](https://colab.research.google.com/github/lohex/DeeplearningExamples/blob/main/projects/bbbc021_morphology_classification/notebooks/01_build_patch_dataset.ipynb)

The notebook offers three enum-based presets:

| Preset | Patches | Plates per class | Wells per class | Intended use |
| --- | ---: | ---: | ---: | --- |
| `SMOKE_TEST` | 1,000 | 1 | 3 | Verify the complete pipeline |
| `DEVELOPMENT` | 10,000 | 2 | 10 | Standard Colab development |
| `FULL` | 200,000 | all | all | Larger training experiment |

## Implementation

Reusable code lives in `src/bbbc021_morphology_classification/pipeline.py`. It owns metadata selection, deterministic split assignment, downloads, channel normalization, foreground-aware patch sampling, shard writing and manifest validation.

Splits are assigned by the plate/well group before any patches are sampled. Every patch retains its source metadata. This prevents direct crop leakage, but patches from the same field are still correlated observations; a future classifier should therefore report field- or well-level performance in addition to patch-level metrics. Plate-level holdouts are the stronger test when enough plates per class are available.

## Outputs

The pipeline writes image/metadata pairs to bounded tar shards and records their split and source identifiers in a manifest. The presets separate a quick end-to-end smoke test from development and full-scale dataset construction, making storage and runtime costs explicit before a large run.

## Local execution

```bash
uv sync --extra bbbc021
jupyter lab projects/bbbc021_morphology_classification/notebooks/01_build_patch_dataset.ipynb
```
