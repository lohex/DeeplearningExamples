# TensorFlow Grad-CAM for time courses

## Goal

Train a one-dimensional convolutional classifier and identify temporal intervals that support individual class predictions.

## Contents

- `notebooks/gradcam.ipynb`: training, held-out evaluation and class-wise Grad-CAM examples
- `src/tensorflow_gradcam/model.py`: classifier architecture
- `src/tensorflow_gradcam/training.py`: function-based Keras training
- `src/tensorflow_gradcam/gradcam.py`: functional attribution and plotting utilities

## Run

```bash
python -m pip install -e ".[tensorflow]"
jupyter lab projects/tensorflow_gradcam/notebooks/gradcam.ipynb
```

Grad-CAM describes model sensitivity. Compare random seeds, target layers and perturbations before drawing a biological conclusion.
