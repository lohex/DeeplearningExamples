# PyTorch time-course classification

## Goal

Classify the TGF-beta stimulation dose of single-cell trajectories while developing a reproducible workflow for temporal ML. The project compares optimization choices and architectural inductive biases instead of treating model tuning as a single undifferentiated search problem.

## ML engineering questions

- How much performance comes from the training strategy, and how much from model capacity or representation?
- Which settings lead to stable convergence, underfitting or overfitting?
- Are first differences, frequency features, multiple temporal scales or long-range attention useful for these trajectories?
- Can model families be compared without using the test fold for iterative decisions?
- Can attention weights and temporal attribution methods identify plausible regions of a time course?
- Which artifacts are required to reload and understand a trained experiment?

## Notebook workflow

| Notebook | Purpose | Engineering question |
| --- | --- | --- |
| [`01_data_visualization.ipynb`](notebooks/01_data_visualization.ipynb) | Train-only trajectories, statistical features and clustering | Which signal characteristics should guide modeling without leaking validation information? |
| [`02_cnn_training.ipynb`](notebooks/02_cnn_training.ipynb) | CNN variants and controlled training ablations | What changes stability and generalization: capacity, learning rate, decay, warmup or regularization? |
| [`02_extended_cnns_training.ipynb`](notebooks/02_extended_cnns_training.ipynb) | Derivative, FFT, gated, multi-scale, TCN, shift-robust and ordinal CNNs | Which domain-motivated temporal representation is useful beyond size and depth? |
| [`02_attention_training.ipynb`](notebooks/02_attention_training.ipynb) | Positional self-attention, CNN attention pooling and hierarchical patch transformers | When is global context useful, and what can attention reveal about the decision? |
| [`03_model_family_comparison.ipynb`](notebooks/03_model_family_comparison.ipynb) | Final CNN, extended-CNN and attention comparison | Does validation-based selection transfer to test performance across accuracy, ranking and class-specific metrics? |

## Implementation

- `src/pytorch_timecourse_classification/models/` contains model definitions and the registry.
- `training.py` provides architecture-independent optimization, schedules, class weighting, balanced sampling and early stopping.
- `experiments.py` keeps repeated notebook orchestration in importable code.
- `tuning.py` supports repeated-seed ablations and Optuna scans.
- `analysis.py` contains common metrics, history comparisons and attention overlays.
- `artifacts.py` saves portable checkpoints, training histories and strategy metadata.

## Evaluation contract

The fixed folds in `Data/tgfb_stimulation_labels.csv` are loaded directly. Preprocessing is fitted on train only. The tune (`validate`) fold controls early stopping, ablations, model selection, classification reports and confusion matrices. The model-development notebooks do not access test data; `03_model_family_comparison.ipynb` opens test only after one candidate per family has been selected.

Accuracy is complemented by class-wise precision and recall, confusion matrices, Average Precision, ROC and precision-recall curves. Training histories retain the selected epoch so convergence and generalization gaps can be inspected rather than reporting only a final score.

## Artifacts

Each experiment stores a rebuildable checkpoint, train/tune history and `metadata.json` under `notebooks/artifacts/<experiment_name>/`. Metadata records model and preprocessing configuration, optimizer, schedule, weight decay, class balancing, random seed and early-stopping strategy. This supports strategy-level reproducibility without claiming bitwise-identical execution across hardware.

## Run

```bash
uv sync --extra torch
jupyter lab projects/pytorch_timecourse_classification/notebooks/
```

For Optuna scans with uv:

```bash
uv sync --extra torch --extra tuning
```
