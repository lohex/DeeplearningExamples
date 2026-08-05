"""Function-based training for the PyTorch time-course classifier."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Subset, TensorDataset


@dataclass(frozen=True, slots=True)
class TrainingConfig:
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
    """Train a classifier and restore the weights with minimum validation loss."""
    cfg = config or TrainingConfig()
    _validate(features, targets, cfg)
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    training_loader, validation_loader = _data_loaders(
        features, targets, cfg, pin_memory=resolved_device.type == "cuda"
    )
    model.to(resolved_device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.learning_rate_decay)

    training_losses: list[float] = []
    validation_losses: list[float] = []
    training_accuracies: list[float] = []
    validation_accuracies: list[float] = []
    best_state = deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_accuracy = _run_epoch(
            model, training_loader, loss_function, resolved_device, optimizer
        )
        validation_loss, validation_accuracy = _run_epoch(
            model, validation_loader, loss_function, resolved_device
        )
        training_losses.append(train_loss)
        validation_losses.append(validation_loss)
        training_accuracies.append(train_accuracy)
        validation_accuracies.append(validation_accuracy)
        scheduler.step()

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % cfg.report_every == 0 or epoch == cfg.epochs:
            print(
                f"epoch {epoch:3d}: train loss {train_loss:.5f}, "
                f"validation loss {validation_loss:.5f}, "
                f"train accuracy {train_accuracy:.1%}, "
                f"validation accuracy {validation_accuracy:.1%}"
            )
        if stale_epochs >= cfg.patience:
            print(f"Early stopping after epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    return TrainingHistory(
        tuple(training_losses),
        tuple(validation_losses),
        tuple(training_accuracies),
        tuple(validation_accuracies),
        best_epoch,
    )


def _validate(features: Tensor, targets: Tensor, config: TrainingConfig) -> None:
    if features.ndim < 2 or targets.ndim != 1 or len(features) != len(targets):
        raise ValueError("features and one-dimensional targets must have equal length.")
    if len(features) < 4 or targets.is_floating_point() or torch.min(targets).item() < 0:
        raise ValueError("targets must be zero-based integers and contain at least four samples.")
    if config.epochs < 1 or config.batch_size < 1 or config.patience < 1:
        raise ValueError("epochs, batch_size and patience must be positive.")
    if not 0.0 < config.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one.")


def _data_loaders(
    features: Tensor,
    targets: Tensor,
    config: TrainingConfig,
    *,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader]:
    indices = np.arange(len(features))
    training_indices, validation_indices = train_test_split(
        indices,
        test_size=config.validation_fraction,
        random_state=config.random_seed,
        stratify=targets.cpu().numpy(),
    )
    dataset = TensorDataset(features.float(), targets.long())
    generator = torch.Generator().manual_seed(config.random_seed)
    return (
        DataLoader(
            Subset(dataset, training_indices.tolist()),
            batch_size=config.batch_size,
            shuffle=True,
            generator=generator,
            pin_memory=pin_memory,
        ),
        DataLoader(
            Subset(dataset, validation_indices.tolist()),
            batch_size=config.batch_size,
            shuffle=False,
            pin_memory=pin_memory,
        ),
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: Optimizer | None = None,
) -> tuple[float, float]:
    learning = optimizer is not None
    model.train(learning)
    total_loss = 0.0
    correct = 0
    sample_count = 0
    for batch_features, batch_targets in loader:
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
        total_loss += loss.item() * len(batch_targets)
        correct += (logits.argmax(1) == batch_targets).sum().item()
        sample_count += len(batch_targets)
    return total_loss / sample_count, correct / sample_count
