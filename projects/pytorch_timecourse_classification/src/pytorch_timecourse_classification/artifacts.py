"""Portable model and training-history artifacts."""

import json
import platform
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import Preprocessor
from .models import build_model
from .training import TrainingConfig, TrainingHistory


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    """Files written for one completed model experiment."""

    model: Path
    history: Path
    metadata: Path


def artifact_root() -> Path:
    """Return the shared ``notebooks/artifacts`` directory."""
    project_dir = Path(__file__).resolve().parents[2]
    return project_dir / "notebooks" / "artifacts"


def artifact_dir_for(experiment_name: str) -> Path:
    """Return ``notebooks/artifacts/<experiment_name>``."""
    if (
        not experiment_name
        or Path(experiment_name).name != experiment_name
        or experiment_name in {".", ".."}
    ):
        raise ValueError("experiment_name must be one non-empty path component.")
    return artifact_root() / experiment_name


def save_experiment_artifacts(
    output_dir: str | Path,
    *,
    model: nn.Module,
    history: TrainingHistory,
    preprocessor: Preprocessor,
    training_config: TrainingConfig | None = None,
    training_targets: torch.Tensor | None = None,
    validation_targets: torch.Tensor | None = None,
    device: torch.device | str | None = None,
) -> ArtifactPaths:
    """Save a rebuildable model, history and human-readable strategy metadata."""
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    model_path = resolved_output_dir / "model.pt"
    history_path = resolved_output_dir / "training_history.npz"
    metadata_path = resolved_output_dir / "metadata.json"

    model_name = getattr(model, "model_name", None)
    model_config = getattr(model, "model_config", None)
    if not isinstance(model_name, str) or not isinstance(model_config, dict):
        raise TypeError("model must expose model_name and model_config.")
    resolved_training_config = training_config or history.config
    if resolved_training_config is None:
        raise ValueError(
            "training_config is required when history does not contain one."
        )

    strategy = {
        "optimizer": "AdamW",
        "scheduler": "StepLR",
        "loss": "CrossEntropyLoss",
        "early_stopping": {
            "monitor": "tune_loss",
            "mode": "min",
            "restore_best_weights": True,
        },
        "class_balance": {
            "strategy": resolved_training_config.class_balance_strategy,
            "weight_formula": (
                "n_samples / (n_classes * class_count)"
                if resolved_training_config.class_balance_strategy
                == "weighted_loss"
                else None
            ),
            "sampler_formula": (
                "sample_weight = 1 / class_count; replacement = true"
                if resolved_training_config.class_balance_strategy
                == "balanced_sampler"
                else None
            ),
        },
    }
    metadata = {
        "artifact_format_version": 2,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "experiment_name": resolved_output_dir.name,
        "model": {
            "name": model_name,
            "class": type(model).__name__,
            "config": model_config,
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
        },
        "training": {
            "config": asdict(resolved_training_config),
            "strategy": strategy,
            "epochs_run": len(history.training_loss),
            "best_epoch": history.best_epoch,
        },
        "data": {
            "folds": {"training": "train", "tuning": "validate"},
            "class_names": preprocessor.class_names,
            "training_class_counts": _class_counts(training_targets),
            "tune_class_counts": _class_counts(validation_targets),
            "preprocessing": {
                "normalization": "global z-score fitted on train only",
                "signal_mean": preprocessor.signal_mean,
                "signal_std": preprocessor.signal_std,
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device) if device is not None else None,
        },
    }

    state_dict = {
        name: parameter.detach().cpu() for name, parameter in model.state_dict().items()
    }
    torch.save(
        {
            "model_name": model_name,
            "model_config": model_config,
            "model_state_dict": state_dict,
            "class_names": preprocessor.class_names,
            "signal_mean": preprocessor.signal_mean,
            "signal_std": preprocessor.signal_std,
            "best_epoch": history.best_epoch,
            "training_config": asdict(resolved_training_config),
            "training_strategy": strategy,
        },
        model_path,
    )
    np.savez_compressed(
        history_path,
        training_loss=np.asarray(history.training_loss),
        validation_loss=np.asarray(history.validation_loss),
        training_accuracy=np.asarray(history.training_accuracy),
        validation_accuracy=np.asarray(history.validation_accuracy),
        best_epoch=np.asarray(history.best_epoch),
    )
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
        metadata_file.write("\n")
    return ArtifactPaths(model_path, history_path, metadata_path)


def load_model(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, dict[str, object]]:
    """Rebuild a model instance and load its saved parameters."""
    resolved_device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
    model = build_model(checkpoint["model_name"], **checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolved_device)
    model.eval()
    return model, checkpoint


def _class_counts(targets: torch.Tensor | None) -> list[int] | None:
    if targets is None:
        return None
    return torch.bincount(targets.detach().cpu().long()).tolist()
