"""Function-based TensorFlow training."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from tensorflow import keras


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 200
    batch_size: int = 256
    validation_fraction: float = 0.25
    patience: int = 20
    random_seed: int = 42
    verbose: int = 1


def train(
    model: keras.Model,
    features: NDArray[np.floating],
    targets: NDArray[np.floating],
    *,
    config: TrainingConfig | None = None,
) -> keras.callbacks.History:
    """Fit a compiled model with early stopping and best-weight restoration."""
    cfg = config or TrainingConfig()
    if len(features) != len(targets):
        raise ValueError("features and targets must have equal length.")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg.patience, restore_best_weights=True
        )
    ]
    return model.fit(
        features,
        targets,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        validation_split=cfg.validation_fraction,
        shuffle=True,
        callbacks=callbacks,
        verbose=cfg.verbose,
    )
