"""Descriptive analysis of trajectory collections."""

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def summarize_dataset(name: str, trajectories: NDArray[np.floating]) -> dict[str, object]:
    """Return compact dimensions and robust value summaries."""
    values = np.asarray(trajectories)
    return {
        "dataset": name,
        "cells": values.shape[0],
        "timepoints": values.shape[1],
        "minimum": float(np.nanmin(values)),
        "median": float(np.nanmedian(values)),
        "maximum": float(np.nanmax(values)),
        "missing_fraction": float(np.isnan(values).mean()),
    }


def trajectory_features(name: str, trajectories: NDArray[np.floating]) -> pd.DataFrame:
    """Calculate interpretable per-cell level, variability and dynamic-range features."""
    values = np.asarray(trajectories)
    baseline = np.median(values[:, : max(3, values.shape[1] // 20)], axis=1)
    return pd.DataFrame(
        {
            "dataset": name,
            "baseline": baseline,
            "mean": np.mean(values, axis=1),
            "standard_deviation": np.std(values, axis=1),
            "dynamic_range": np.max(values, axis=1) - np.min(values, axis=1),
            "maximum_absolute_step": np.max(np.abs(np.diff(values, axis=1)), axis=1),
        }
    )
