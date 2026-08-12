"""Shared visual analysis of model training histories."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from torch import Tensor

from .training import TrainingHistory


def summarize_history(history: TrainingHistory) -> dict[str, int | float]:
    """Return metrics at the checkpoint selected by validation loss."""
    best_index = history.best_epoch - 1
    return {
        "epochs_run": len(history.training_loss),
        "best_epoch": history.best_epoch,
        "training_loss": history.training_loss[best_index],
        "tune_loss": history.validation_loss[best_index],
        "training_accuracy": history.training_accuracy[best_index],
        "tune_accuracy": history.validation_accuracy[best_index],
        "accuracy_gap": (
            history.training_accuracy[best_index]
            - history.validation_accuracy[best_index]
        ),
    }


def plot_training_history(history: TrainingHistory) -> Figure:
    """Plot loss and accuracy with the restored epoch marked."""
    epochs = np.arange(1, len(history.training_loss) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(epochs, history.training_loss, label="train")
    axes[0].plot(epochs, history.validation_loss, label="tune")
    axes[0].set(title="Loss", xlabel="epoch")
    axes[1].plot(epochs, history.training_accuracy, label="train")
    axes[1].plot(epochs, history.validation_accuracy, label="tune")
    axes[1].set(title="Accuracy", xlabel="epoch")
    for axis in axes:
        axis.axvline(
            history.best_epoch,
            color="black",
            linestyle=":",
            label="restored epoch",
        )
        axis.legend()
        axis.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def plot_history_comparison(
    histories: dict[str, TrainingHistory],
) -> Figure:
    """Compare train/tune loss and accuracy with restored-epoch dots."""
    if not histories:
        raise ValueError("histories must contain at least one experiment.")
    panels = (
        ("training_loss", "Train loss", "loss"),
        ("validation_loss", "Tune loss", "loss"),
        ("training_accuracy", "Train accuracy", "accuracy"),
        ("validation_accuracy", "Tune accuracy", "accuracy"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = dict(zip(histories, plt.get_cmap("tab10").colors))
    for axis, (attribute, title, ylabel) in zip(axes.flat, panels):
        for experiment_name, history in histories.items():
            values = np.asarray(getattr(history, attribute))
            epochs = np.arange(1, len(values) + 1)
            color = colors[experiment_name]
            axis.plot(epochs, values, color=color, label=experiment_name)
            axis.scatter(
                history.best_epoch,
                values[history.best_epoch - 1],
                color=color,
                edgecolor="black",
                s=55,
                zorder=3,
            )
        axis.set(title=title, xlabel="epoch", ylabel=ylabel)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("Training-history comparison")
    figure.tight_layout()
    return figure


def attention_rollout(attention: Tensor, *, pooling: str) -> Tensor:
    """Project multi-layer self-attention back to input timepoints.

    Heads are averaged and an identity residual is added before multiplying
    attention matrices across layers. The result is normalized per sample.
    """
    if attention.ndim != 5 or attention.shape[-1] != attention.shape[-2]:
        raise ValueError(
            "attention must have shape (batch, layers, heads, tokens, tokens)."
        )
    if pooling not in {"mean", "cls"}:
        raise ValueError("pooling must be 'mean' or 'cls'.")

    mean_attention = attention.mean(dim=2)
    token_count = mean_attention.shape[-1]
    identity = torch.eye(
        token_count,
        dtype=mean_attention.dtype,
        device=mean_attention.device,
    ).expand(len(mean_attention), -1, -1)
    joint_attention = identity
    for layer_attention in mean_attention.unbind(dim=1):
        layer_attention = layer_attention + identity
        layer_attention = layer_attention / layer_attention.sum(
            dim=-1, keepdim=True
        )
        joint_attention = layer_attention @ joint_attention

    if pooling == "cls":
        importance = joint_attention[:, 0, 1:]
    else:
        importance = joint_attention.mean(dim=1)
    return importance / importance.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def plot_attention_overlay(
    trajectory: np.ndarray,
    importance: np.ndarray,
    *,
    axis: plt.Axes,
    title: str,
) -> None:
    """Plot one trajectory with timepoints colored by rollout importance."""
    if trajectory.ndim != 1 or importance.shape != trajectory.shape:
        raise ValueError("trajectory and importance must be equal-length vectors.")
    time = np.arange(len(trajectory))
    axis.plot(time, trajectory, color="0.65", linewidth=1, zorder=1)
    points = axis.scatter(
        time,
        trajectory,
        c=importance,
        cmap="magma",
        s=10,
        zorder=2,
    )
    axis.set(title=title, xlabel="time index", ylabel="signal")
    axis.figure.colorbar(points, ax=axis, label="attention rollout")
