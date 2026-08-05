# DeepLearningExamples

Small, self-contained deep-learning projects for biological image and time-course data. Notebooks document experiments; reusable model, training and pipeline code lives in importable Python packages.

## Repository structure

```text
Data/                         Example time-course datasets
src/deeplearning_examples/    Shared data loading and preparation
projects/
  <project>/
    README.md                  Goal, method and execution notes
    notebooks/                Experiment orchestration and interpretation
    src/<package>/             Project-specific implementation
```

## Projects

| Project | Question | Framework |
| --- | --- | --- |
| [PyTorch CDGAN](projects/pytorch_cdgan/) | Can a GAN reproduce the distribution of stimulated trajectories? | PyTorch |
| [PyTorch time-course classification](projects/pytorch_timecourse_classification/) | Can raw signal levels and temporal differences distinguish SMAD conditions? | PyTorch |
| [TensorFlow attention](projects/tensorflow_attention/) | Does cross-attention between levels and changes improve classification? | TensorFlow |
| [TensorFlow Grad-CAM](projects/tensorflow_gradcam/) | Which temporal intervals support a classifier prediction? | TensorFlow |
| [Dataset visualization](projects/dataset_visualization/) | What are the dimensions, ranges and dynamics of the supplied data? | NumPy, pandas |
| [BBBC021 morphology classification](projects/bbbc021_morphology_classification/) | Can treatment-specific cell morphology be classified and interpreted with Grad-CAM? | Pillow, pandas |

## Installation

Install once from the repository root. Editable installation makes both shared and project-specific packages importable without modifying `sys.path`.

```bash
python -m pip install -e .
python -m pip install -e ".[torch]"
python -m pip install -e ".[tensorflow]"
python -m pip install -e ".[bbbc021]"
```

Use `python -m pip install -e ".[all]"` to install all optional dependencies.

## Design conventions

- Notebooks contain data selection, experiment configuration, plots and interpretation.
- Neural-network definitions live in each project's `model.py`.
- Optimization lives in `training.py` and is exposed as a function, not a trainer class.
- Shared dataset imports and representations live in `src/deeplearning_examples/`.
- Train, validation and test splits should follow the biological experimental unit rather than the number of derived samples.
