# PyTorch CDGAN

## Goal

Learn a generative distribution of stimulated single-cell signaling trajectories and inspect whether generated samples reproduce temporal diversity rather than only an average shape.

## Contents

- `notebooks/cdgan.ipynb`: data inspection, experiment configuration, diagnostics and interpretation
- `src/pytorch_cdgan/model.py`: generator and discriminator
- `src/pytorch_cdgan/training.py`: function-based adversarial training loop

## Run

```bash
python -m pip install -e ".[torch]"
jupyter lab projects/pytorch_cdgan/notebooks/cdgan.ipynb
```

The notebook loads only the fixed train fold from `Data/tgfb_stimulation_time_courses.npy` and `Data/tgfb_stimulation_labels.csv`. GAN scores near 0.5 are not sufficient evidence of quality; inspect diversity, nearest neighbors and biological trajectory features as well.
