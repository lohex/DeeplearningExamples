"""Data import utilities shared by the example notebooks."""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray


FloatArray = NDArray[np.floating]
LabelArray = NDArray[np.object_]
Fold = Literal["train", "validate", "test"]
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "Data"
FOLD_NAMES = frozenset({"train", "validate", "test"})


class DataImportError(RuntimeError):
    """Raised when an example dataset cannot be imported."""


def load_data(
    fold: Fold,
    data_dir: str | Path | None = None,
) -> tuple[FloatArray, LabelArray, NDArray[np.integer]]:
    """Load time courses and labels for one fixed dataset fold.

    ``fold`` must be ``"train"``, ``"validate"`` or ``"test"``. Returns
    ``(time_courses, doses, years)`` for that fold.
    """
    if fold not in FOLD_NAMES:
        raise ValueError(f"fold must be one of {sorted(FOLD_NAMES)}, got {fold!r}")

    resolved_data_dir = _resolve_data_dir(data_dir)
    _validate_data_dir(resolved_data_dir)

    timecourses_path = resolved_data_dir / "tgfb_stimulation_time_courses.npy"
    labels_path = resolved_data_dir / "tgfb_stimulation_labels.csv"

    try:
        time_courses = np.load(timecourses_path, allow_pickle=False)
    except (FileNotFoundError, OSError, ValueError) as error:
        raise DataImportError(
            f"Could not read timecourses file: {timecourses_path}"
        ) from error

    try:
        labels = pd.read_csv(labels_path)
    except (FileNotFoundError, OSError, pd.errors.ParserError) as error:
        raise DataImportError(f"Could not read labels file: {labels_path}") from error

    required_columns = {"dose", "year", "fold"}
    missing_columns = required_columns.difference(labels.columns)
    if missing_columns:
        raise DataImportError(
            f"Labels file missing required columns {sorted(missing_columns)}: "
            f"{labels_path}"
        )

    doses = labels["dose"].to_numpy(dtype=object)
    years = labels["year"].to_numpy(dtype=int)
    folds = labels["fold"].to_numpy(dtype=object)

    observed_folds = set(folds)
    missing_folds = FOLD_NAMES.difference(observed_folds)
    unexpected_folds = observed_folds.difference(FOLD_NAMES)
    if missing_folds or unexpected_folds:
        raise DataImportError(
            "Labels file must contain exactly the folds "
            f"{sorted(FOLD_NAMES)}; missing {sorted(missing_folds)}, "
            f"unexpected {sorted(unexpected_folds)}: {labels_path}"
        )

    # Ensure samples are the first axis: prefer shape (n_samples, n_timepoints,...)
    if time_courses.shape[0] != doses.shape[0]:
        # If the last axis matches, transpose to bring samples to axis 0
        if time_courses.shape[-1] == doses.shape[0]:
            time_courses = np.asarray(time_courses).T
        else:
            raise DataImportError("Number of timecourses does not match number of labels")

    selected = folds == fold
    return time_courses[selected], doses[selected], years[selected]


def _resolve_data_dir(data_dir: str | Path | None) -> Path:
    if data_dir is None:
        return DEFAULT_DATA_DIR
    return Path(data_dir).expanduser().resolve()


def _validate_data_dir(data_dir: Path) -> None:
    if not data_dir.is_dir():
        raise DataImportError(f"Data directory does not exist: {data_dir}")
