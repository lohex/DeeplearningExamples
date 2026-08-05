"""Shared preparation of the SMAD classification dataset."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .io import FloatArray, load_smad_timecourses


IntArray = NDArray[np.integer]
CLASS_NAMES = ("control", "TGF-beta high", "GDF11 high")


@dataclass(frozen=True, slots=True)
class SmadClassificationData:
    """Trajectories and zero-based labels shared by classifier projects."""

    times: FloatArray
    trajectories: FloatArray
    targets: IntArray
    class_names: tuple[str, ...] = CLASS_NAMES

    @property
    def channels_first(self) -> FloatArray:
        """Return PyTorch layout: samples, channels, timepoints."""
        return self.trajectories[:, np.newaxis, :]

    @property
    def channels_last(self) -> FloatArray:
        """Return TensorFlow layout: samples, timepoints, channels."""
        return self.trajectories[..., np.newaxis]


def load_smad_classification_data(
    data_dir: str | Path | None = None,
) -> SmadClassificationData:
    """Load and combine the three conditions used for classification."""
    source = load_smad_timecourses(data_dir)
    conditions = (source.control, source.tgfb_high, source.gdf11_high)
    trajectories = np.concatenate(conditions, axis=0)
    targets = np.concatenate(
        [np.full(len(values), index, dtype=np.int64) for index, values in enumerate(conditions)]
    )
    return SmadClassificationData(
        times=source.times,
        trajectories=trajectories,
        targets=targets,
    )
