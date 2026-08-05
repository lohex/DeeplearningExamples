"""Generator and discriminator for fixed-length signaling trajectories."""

import torch
from torch import Tensor, nn


class Discriminator(nn.Module):
    def __init__(self, input_length: int = 289, dropout: float = 0.3) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_length, input_length * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(input_length * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, trajectories: Tensor) -> Tensor:
        return self.network(trajectories)


class Generator(nn.Module):
    def __init__(self, noise_size: int = 16, output_length: int = 289) -> None:
        super().__init__()
        self.noise_size = noise_size
        self.network = nn.Sequential(
            nn.Linear(noise_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_length),
            nn.Unflatten(1, (1, output_length)),
            nn.Conv1d(1, 32, 5, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 16, 7, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 1, 11, padding="same"),
        )

    def forward(self, noise: Tensor) -> Tensor:
        return self.network(noise)


def initialize_weights(module: nn.Module) -> None:
    """Initialize linear, convolutional and batch-normalization layers."""
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.normal_(module.weight.data, 0.0, 0.5)
        if module.bias is not None:
            nn.init.zeros_(module.bias.data)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.normal_(module.weight.data, 1.0, 0.5)
        nn.init.zeros_(module.bias.data)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
