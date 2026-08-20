# DeepLearningExamples

Small, self-contained PyTorch projects for biological image and time-course data. The repository develops the workflows from data inspection and fixed splits through reproducible training, tuning, interpretation, model comparison and generative modeling. Notebooks document questions and results; reusable implementation lives in importable Python packages.

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

| Project | Scientific question | ML engineering focus |
| --- | --- | --- |
| [Time-course classification](projects/pytorch_timecourse_classification/) | Which temporal representations distinguish TGF-beta stimulation doses? | Controlled ablations, optimization versus architecture, reproducible model selection and temporal explanations |
| [Conditional GAN](projects/pytorch_cdgan/) | Can dose-conditioned generators reproduce realistic single-cell trajectories? | Stable adversarial training, conditional generation and quality diagnostics beyond loss curves |
| [BBBC021 morphology classification](projects/bbbc021_morphology_classification/) | How can microscopy morphology be prepared for treatment classification? | Group-aware splitting, scalable dataset construction and protection against biological leakage |

## ML engineering questions

The projects are organized around questions that recur in real ML systems:

- **Data validity:** What is the independent experimental unit, and how are train, validation and test splits fixed before modeling to prevent leakage?
- **Training or model?** When performance changes, is the cause optimization, regularization and sampling, or the architecture and its inductive bias?
- **Reliability:** Is a result stable across seeds, does training converge, and do the histories indicate underfitting, overfitting or unstable optimization?
- **Search strategy:** Which hypotheses deserve controlled ablations, and when is a broader Optuna search more appropriate?
- **Evaluation discipline:** How are models selected without repeatedly consulting the test fold, and which metrics expose class-specific errors hidden by accuracy?
- **Interpretability:** Does a model use plausible temporal or morphological evidence, and do attention or attribution maps support that conclusion?
- **Reproducibility:** Which preprocessing state, model configuration, training strategy and checkpoint must be saved to rebuild an experiment?
- **Generative quality:** Do synthetic trajectories preserve dose-dependent distributions, dynamics and diversity rather than merely fooling a discriminator?

Together, the projects move from trustworthy dataset construction through discriminative model development to conditional generation. A well-evaluated classifier can later also support the GAN project as an independent test of whether generated trajectories retain dose information.

### Time-course workflow

The classification project contains the main end-to-end engineering example:

1. [`01_data_visualization.ipynb`](projects/pytorch_timecourse_classification/notebooks/01_data_visualization.ipynb) establishes hypotheses from only the fixed training fold, statistical trajectory features and dose-dependent clusters.
2. [`02_cnn_training.ipynb`](projects/pytorch_timecourse_classification/notebooks/02_cnn_training.ipynb) asks how capacity and training hyperparameters affect stability, convergence and over- or underfitting.
3. [`02_extended_cnns_training.ipynb`](projects/pytorch_timecourse_classification/notebooks/02_extended_cnns_training.ipynb) tests whether targeted temporal inductive biases add more value than simply scaling a CNN.
4. [`02_attention_training.ipynb`](projects/pytorch_timecourse_classification/notebooks/02_attention_training.ipynb) studies long-range context, positional information and the limits of attention as an explanation.
5. [`03_model_family_comparison.ipynb`](projects/pytorch_timecourse_classification/notebooks/03_model_family_comparison.ipynb) separates validation-based model selection from the final test comparison and broadens evaluation beyond accuracy.

The GAN notebook is [`projects/pytorch_cdgan/notebooks/cdgan.ipynb`](projects/pytorch_cdgan/notebooks/cdgan.ipynb). The BBBC021 project currently contains the dataset-construction notebook documented in its project README.

## Installation

The repository uses `uv`. Install the PyTorch projects from the repository root with:

```bash
uv sync --extra torch
```

Add optional experiment tuning or image-dataset dependencies as needed:

```bash
uv sync --extra torch --extra tuning
uv sync --extra bbbc021
uv sync --all-extras
```

The equivalent editable pip installation is `python -m pip install -e ".[torch,tuning]"`.

## Design conventions

- Notebooks contain data selection, experiment configuration, plots and interpretation.
- Neural-network definitions live in each project's `model.py` or `models/` package.
- Optimization lives in `training.py` and is exposed as a function, not a trainer class.
- Shared experiment orchestration, artifact persistence and analysis live in project `src/` packages rather than notebook helper functions.
- Shared dataset imports and representations live in `src/deeplearning_examples/`.
- Fixed train, validation and test assignments are stored with the data; preprocessing is fitted on train only and test remains untouched until final evaluation.
- Rebuildable checkpoints are accompanied by training histories and strategy metadata.
