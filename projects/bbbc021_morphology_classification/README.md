# BBBC021 morphology classification

## Goal

Develop an image-classification and Grad-CAM workflow for five interpretable BBBC021 treatment classes: DMSO, cytochalasin B, nocodazole, taxol and AZ-A.

## Notebooks

1. [`01_build_patch_dataset.ipynb`](notebooks/01_build_patch_dataset.ipynb): select source wells, download the required plates, estimate storage and generate balanced WebDataset shards.

[Open notebook 01 in Colab](https://colab.research.google.com/github/lohex/DeeplearningExamples/blob/main/projects/bbbc021_morphology_classification/notebooks/01_build_patch_dataset.ipynb)

The notebook offers three enum-based presets:

| Preset | Patches | Plates per class | Wells per class | Intended use |
| --- | ---: | ---: | ---: | --- |
| `SMOKE_TEST` | 1,000 | 1 | 3 | Verify the complete pipeline |
| `DEVELOPMENT` | 10,000 | 2 | 10 | Standard Colab development |
| `FULL` | 200,000 | all | all | Larger training experiment |

## Implementation

Reusable code lives in `src/bbbc021_morphology_classification/pipeline.py`. Splits are assigned by plate and well before patch extraction. Derived patches from one field remain correlated, so evaluation should also be aggregated by field or well.

## Local execution

```bash
python -m pip install -e ".[bbbc021]"
jupyter lab projects/bbbc021_morphology_classification/notebooks/01_build_patch_dataset.ipynb
```
