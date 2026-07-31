"""Functional PyTorch training utilities for the example notebooks."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Subset, TensorDataset


class TorchTrainingError(ValueError):
    """Raised when a PyTorch training configuration or input is invalid."""


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Hyperparameters controlling optimization and early stopping."""

    epochs: int = 100
    batch_size: int = 512
    validation_fraction: float = 0.25
    learning_rate: float = 1e-3
    learning_rate_decay: float = 0.92
    patience: int = 10
    report_every: int = 5
    random_seed: int = 42


@dataclass(frozen=True, slots=True)
class TrainingHistory:
    """Metrics collected during training after restoring the best model."""

    training_loss: tuple[float, ...]
    validation_loss: tuple[float, ...]
    training_accuracy: tuple[float, ...]
    validation_accuracy: tuple[float, ...]
    best_epoch: int


def train(
    model: nn.Module,
    features: Tensor,
    targets: Tensor,
    *,
    config: TrainingConfig | None = None,
    device: torch.device | str | None = None,
) -> TrainingHistory:
    """Train a classifier and restore the state with minimum validation loss.

    Targets must be zero-based integer class indices. The model must return
    logits because ``CrossEntropyLoss`` applies log-softmax internally.
    """
    resolved_config = config if config is not None else TrainingConfig()
    _validate_training_inputs(model, features, targets, resolved_config)

    resolved_device = _resolve_device(device)
    training_loader, validation_loader = _create_data_loaders(
        features,
        targets,
        resolved_config,
        use_pinned_memory=resolved_device.type == "cuda",
    )

    model.to(resolved_device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=resolved_config.learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=resolved_config.learning_rate_decay,
    )

    training_losses: list[float] = []
    validation_losses: list[float] = []
    training_accuracies: list[float] = []
    validation_accuracies: list[float] = []
    best_state = deepcopy(model.state_dict())
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, resolved_config.epochs + 1):
        training_loss, training_accuracy = _run_epoch(
            model,
            training_loader,
            loss_function,
            resolved_device,
            optimizer=optimizer,
        )
        validation_loss, validation_accuracy = _run_epoch(
            model,
            validation_loader,
            loss_function,
            resolved_device,
        )

        training_losses.append(training_loss)
        validation_losses.append(validation_loss)
        training_accuracies.append(training_accuracy)
        validation_accuracies.append(validation_accuracy)
        scheduler.step()

        improved = validation_loss < best_validation_loss
        if improved:
            best_validation_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        should_report = (
            epoch == 1
            or epoch % resolved_config.report_every == 0
            or epoch == resolved_config.epochs
        )
        if should_report:
            _report_epoch(
                epoch,
                training_loss,
                validation_loss,
                training_accuracy,
                validation_accuracy,
            )

        if epochs_without_improvement >= resolved_config.patience:
            print(f"Early stopping after epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    return TrainingHistory(
        training_loss=tuple(training_losses),
        validation_loss=tuple(validation_losses),
        training_accuracy=tuple(training_accuracies),
        validation_accuracy=tuple(validation_accuracies),
        best_epoch=best_epoch,
    )


def _validate_training_inputs(
    model: nn.Module,
    features: Tensor,
    targets: Tensor,
    config: TrainingConfig,
) -> None:
    if not isinstance(model, nn.Module):
        raise TorchTrainingError("model must be a torch.nn.Module.")
    if features.ndim < 2:
        raise TorchTrainingError("features must include batch and feature axes.")
    if targets.ndim != 1:
        raise TorchTrainingError("targets must be a one-dimensional tensor.")
    if len(features) != len(targets):
        raise TorchTrainingError("features and targets must have equal length.")
    if len(features) < 4:
        raise TorchTrainingError("At least four samples are required.")
    if targets.is_floating_point():
        raise TorchTrainingError("targets must contain integer class indices.")
    if torch.min(targets).item() < 0:
        raise TorchTrainingError("targets must be zero-based non-negative indices.")
    if config.epochs < 1:
        raise TorchTrainingError("epochs must be positive.")
    if config.batch_size < 1:
        raise TorchTrainingError("batch_size must be positive.")
    if not 0.0 < config.validation_fraction < 1.0:
        raise TorchTrainingError("validation_fraction must be between 0 and 1.")
    if config.learning_rate <= 0.0:
        raise TorchTrainingError("learning_rate must be positive.")
    if not 0.0 < config.learning_rate_decay <= 1.0:
        raise TorchTrainingError("learning_rate_decay must be in (0, 1].")
    if config.patience < 1:
        raise TorchTrainingError("patience must be positive.")
    if config.report_every < 1:
        raise TorchTrainingError("report_every must be positive.")

    class_values, class_counts = torch.unique(targets, return_counts=True)
    class_count = len(class_values)
    validation_count = int(np.ceil(len(targets) * config.validation_fraction))
    training_count = len(targets) - validation_count
    if class_count < 2:
        raise TorchTrainingError("targets must contain at least two classes.")
    if torch.min(class_counts).item() < 2:
        raise TorchTrainingError("Each class must contain at least two samples.")
    if min(training_count, validation_count) < class_count:
        raise TorchTrainingError(
            "The split must allocate at least one sample per class to both subsets."
        )


def _resolve_device(device: torch.device | str | None) -> torch.device:
    if device is not None:
        return torch.device(device)

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_name)


def _create_data_loaders(
    features: Tensor,
    targets: Tensor,
    config: TrainingConfig,
    *,
    use_pinned_memory: bool,
) -> tuple[DataLoader, DataLoader]:
    indices = np.arange(len(features))
    target_values = targets.detach().cpu().numpy()
    training_indices, validation_indices = train_test_split(
        indices,
        test_size=config.validation_fraction,
        random_state=config.random_seed,
        stratify=target_values,
    )

    dataset = TensorDataset(features.float(), targets.long())
    training_subset = Subset(dataset, training_indices.tolist())
    validation_subset = Subset(dataset, validation_indices.tolist())
    generator = torch.Generator().manual_seed(config.random_seed)

    training_loader = DataLoader(
        training_subset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=use_pinned_memory,
    )
    validation_loader = DataLoader(
        validation_subset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=use_pinned_memory,
    )
    return training_loader, validation_loader


def _run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    *,
    optimizer: Optimizer | None = None,
) -> tuple[float, float]:
    learning = optimizer is not None
    model.train(mode=learning)
    total_loss = 0.0
    correct_predictions = 0
    sample_count = 0

    for batch_features, batch_targets in data_loader:
        batch_features = batch_features.to(device, non_blocking=True)
        batch_targets = batch_targets.to(device, non_blocking=True)

        if learning:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(learning):
            logits = model(batch_features)
            loss = loss_function(logits, batch_targets)

        if learning:
            loss.backward()
            optimizer.step()

        batch_size = len(batch_targets)
        total_loss += loss.item() * batch_size
        correct_predictions += (
            logits.argmax(dim=1) == batch_targets
        ).sum().item()
        sample_count += batch_size

    mean_loss = total_loss / sample_count
    accuracy = correct_predictions / sample_count
    return mean_loss, accuracy


def _report_epoch(
    epoch: int,
    training_loss: float,
    validation_loss: float,
    training_accuracy: float,
    validation_accuracy: float,
) -> None:
    print(
        f"epoch {epoch:3d}: "
        f"train loss {training_loss:.5f}, "
        f"validation loss {validation_loss:.5f}, "
        f"train accuracy {training_accuracy:.1%}, "
        f"validation accuracy {validation_accuracy:.1%}"
    )
