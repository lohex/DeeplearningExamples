"""Function-based training for the Grad-CAM classifier."""

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
    verbose: int = 1


def train(
    model: keras.Model,
    features: NDArray[np.floating],
    targets: NDArray[np.floating],
    *,
    config: TrainingConfig | None = None,
) -> keras.callbacks.History:
    cfg = config or TrainingConfig()
    return model.fit(
        features,
        targets,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        validation_split=cfg.validation_fraction,
        shuffle=True,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=cfg.patience, restore_best_weights=True
            )
        ],
        verbose=cfg.verbose,
    )
