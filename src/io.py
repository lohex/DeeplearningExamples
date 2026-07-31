"""Data import utilities shared by the example notebooks."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.io import loadmat


FloatArray = NDArray[np.floating]
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "Data"


class DataImportError(RuntimeError):
    """Raised when an example dataset cannot be imported."""


@dataclass(frozen=True, slots=True)
class SmadTimecourseData:
    """SMAD trajectories with one row per cell and one column per timepoint."""

    times: FloatArray
    control: FloatArray
    tgfb_high: FloatArray
    gdf11_high: FloatArray


def load_smad_timecourses(
    data_dir: str | Path | None = None,
) -> SmadTimecourseData:
    """Load the three SMAD conditions used by the classification notebooks."""
    resolved_data_dir = _resolve_data_dir(data_dir)
    _validate_data_dir(resolved_data_dir)

    times, control = _load_timecourse_csv(
        resolved_data_dir / "smad_timecourse_singlecells_ctl.csv"
    )
    tgfb_times, tgfb_high = _load_timecourse_csv(
        resolved_data_dir / "smad_timecourse_singlecells_stv_tgfb_high.csv"
    )
    gdf11_times, gdf11_high = _load_timecourse_csv(
        resolved_data_dir / "smad_timecourse_singlecells_gdf11_high.csv"
    )

    _validate_matching_times(times, tgfb_times, "TGF-beta")
    _validate_matching_times(times, gdf11_times, "GDF11")

    return SmadTimecourseData(
        times=times,
        control=control,
        tgfb_high=tgfb_high,
        gdf11_high=gdf11_high,
    )


def load_stimulation_timecourses(
    data_dir: str | Path | None = None,
) -> FloatArray:
    """Load and combine the two stimulation datasets used by the CDGAN."""
    resolved_data_dir = _resolve_data_dir(data_dir)
    _validate_data_dir(resolved_data_dir)

    first_data = _load_matlab_trajectories(
        resolved_data_dir / "Stimulation_100pM_2013.mat"
    )
    second_data = _load_matlab_trajectories(
        resolved_data_dir / "Stimulation_100pM_2014.mat"
    )
    return np.concatenate((first_data, second_data), axis=0)


def _resolve_data_dir(data_dir: str | Path | None) -> Path:
    if data_dir is None:
        return DEFAULT_DATA_DIR
    return Path(data_dir).expanduser().resolve()


def _validate_data_dir(data_dir: Path) -> None:
    if not data_dir.is_dir():
        raise DataImportError(f"Data directory does not exist: {data_dir}")


def _load_timecourse_csv(path: Path) -> tuple[FloatArray, FloatArray]:
    try:
        frame = pd.read_csv(path)
    except (FileNotFoundError, OSError, pd.errors.ParserError) as error:
        raise DataImportError(f"Could not read timecourse file: {path}") from error

    if "time" not in frame.columns:
        raise DataImportError(f"Missing 'time' column in: {path}")

    times = frame["time"].to_numpy(dtype=np.float64)
    trajectories = frame.drop(columns="time").to_numpy(dtype=np.float64).T
    return times, trajectories


def _load_matlab_trajectories(path: Path) -> FloatArray:
    try:
        matlab_data = loadmat(path)
    except (FileNotFoundError, OSError, ValueError) as error:
        raise DataImportError(f"Could not read MATLAB file: {path}") from error

    if "DataOI_r" not in matlab_data:
        raise DataImportError(f"Missing 'DataOI_r' array in: {path}")

    return np.asarray(matlab_data["DataOI_r"].T, dtype=np.float64)


def _validate_matching_times(
    reference: FloatArray,
    candidate: FloatArray,
    condition: str,
) -> None:
    matching_shape = reference.shape == candidate.shape
    matching_values = matching_shape and np.allclose(reference, candidate)
    if not matching_values:
        raise DataImportError(
            f"Timepoints for {condition} do not match the control condition."
        )
