"""TensorFlow attention model for biological time courses."""

from .model import build_attention_model, build_baseline_model
from .training import TrainingConfig, train

__all__ = ["build_attention_model", "build_baseline_model", "TrainingConfig", "train"]
