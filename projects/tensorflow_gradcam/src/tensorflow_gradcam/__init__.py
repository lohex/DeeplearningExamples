"""TensorFlow Grad-CAM example for one-dimensional time courses."""

from .gradcam import compute_gradcam, plot_gradcam
from .model import build_model
from .training import TrainingConfig, train

__all__ = ["build_model", "TrainingConfig", "train", "compute_gradcam", "plot_gradcam"]
