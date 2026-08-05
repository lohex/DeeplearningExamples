# PyTorch time-course classification

## Goal

Classify control, high TGF-beta and high GDF11 single-cell trajectories. Separate convolutional branches represent the raw signal and its first-order difference.

## Contents

- `notebooks/timecourse_classification.ipynb`: complete experiment and held-out evaluation
- `src/pytorch_timecourse_classification/model.py`: classifier architecture
- `src/pytorch_timecourse_classification/training.py`: `train` function with stratified validation and early stopping

## Run

```bash
python -m pip install -e ".[torch]"
jupyter lab projects/pytorch_timecourse_classification/notebooks/timecourse_classification.ipynb
```

The notebook keeps an external test split while the training function derives a validation split from the remaining cells.
