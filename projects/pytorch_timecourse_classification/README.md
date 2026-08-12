# PyTorch time-course classification

## Goal

Classify the TGF-beta stimulation dose of single-cell trajectories. Separate convolutional branches represent the raw signal and its first-order difference.

## Contents

- `notebooks/01_data_visualization.ipynb`: training-fold trajectory and feature exploration
- `notebooks/02_cnn_training.ipynb`: comparison of compact, baseline and wide CNNs
- `notebooks/02_extended_cnns_training.ipynb`: first-difference, FFT, gated-fusion, multi-scale, TCN, shift-robust and ordinal CNNs
- `notebooks/02_attention_training.ipynb`: three positional self-attention variants, CNN attention pooling and a hierarchical patch transformer
- `notebooks/03_model_family_comparison.ipynb`: tune-based family selection followed by one final test comparison with ROC/PR analysis
- `src/pytorch_timecourse_classification/models/`: model definitions and registry
- `src/pytorch_timecourse_classification/training.py`: architecture-independent training with fixed-fold validation
- `src/pytorch_timecourse_classification/artifacts.py`: portable checkpoints, training histories and strategy metadata

## Run

```bash
python -m pip install -e ".[torch]"
jupyter lab projects/pytorch_timecourse_classification/notebooks/
```

The folds stored in `Data/tgfb_stimulation_labels.csv` are loaded directly. The tune (`validate`) fold controls early stopping and supplies classification reports and confusion matrices. The model notebooks do not access the test fold.

Each experiment stores a rebuildable checkpoint, train/tune history and `metadata.json` under `notebooks/artifacts/<experiment_name>/`, for example `notebooks/artifacts/cnn_compact/`. The metadata records model hyperparameters, preprocessing, optimizer, learning-rate schedule, weight decay, class balancing and early stopping.
