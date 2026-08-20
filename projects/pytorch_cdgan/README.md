# PyTorch CDGAN

## Goal

Learn dose-conditioned distributions of single-cell signaling trajectories. The central challenge is not only producing visually plausible curves, but preserving dose-specific dynamics, variability and roughness without collapsing to an average trajectory.

## ML engineering questions

- How can adversarial training be diagnosed when losses and discriminator scores alone do not measure sample quality?
- Does generating a full-resolution signal directly differ from predicting coarse control points and learning the upsampling corrections?
- How should layer initialization depend on fan-in so activation scales remain controlled at the start of training?
- Does conditioning both generator and discriminator produce samples with the requested dose characteristics?
- Which distributional and temporal statistics reveal excessive noise, oversmoothing or mode collapse?

## Model design

The generator concatenates Gaussian latent noise with a learned dose embedding. Two variants are compared:

1. **Full-resolution generator:** a dense projection creates the complete trajectory before convolutional refinement.
2. **Coarse generator:** a configurable number of control points is predicted, linearly upsampled to the original length and refined by temporal convolutions.

The discriminator receives the trajectory together with a dose embedding broadcast over time. Linear and convolutional layers use fan-in-aware Kaiming initialization instead of a fixed weight standard deviation; this prevents variance from compounding through layers and creating unrealistically large initial signals.

## Contents

- [`notebooks/cdgan.ipynb`](notebooks/cdgan.ipynb): data inspection, both generator variants, per-variant training diagnostics, dose-wise sample analysis and final comparison
- `src/pytorch_cdgan/model.py`: generator and discriminator
- `src/pytorch_cdgan/training.py`: function-based adversarial training loop

## Evaluation

The notebook trains only on the fixed train fold. It compares real data and generated samples separately for every dose using example trajectories and temporal features such as mean, standard deviation, maximum, dynamic range, endpoint change and mean absolute step. Snapshot plots expose how samples evolve during training. The final comparison places real data, the full-resolution generator and the coarse generator side by side.

Discriminator outputs near 0.5 are only a training diagnostic. They are not sufficient evidence of realism, diversity or correct conditioning; feature distributions and dose-wise samples remain essential.

## Run

```bash
uv sync --extra torch
jupyter lab projects/pytorch_cdgan/notebooks/cdgan.ipynb
```
