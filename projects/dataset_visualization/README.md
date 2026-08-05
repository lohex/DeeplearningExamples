# Dataset visualization

## Goal

Inventory the supplied time-course datasets and compare cell counts, signal ranges, population summaries and interpretable trajectory features before modeling.

## Contents

- `notebooks/visualize_datasets.ipynb`: tables and figures for both dataset families
- `src/dataset_visualization/analysis.py`: reusable summary and feature functions

## Run

```bash
python -m pip install -e .
jupyter lab projects/dataset_visualization/notebooks/visualize_datasets.ipynb
```

The feature summaries are descriptive. They should not be treated as labels or independent biological observations.
