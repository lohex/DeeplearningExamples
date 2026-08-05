"""Neural network for classifying single-cell signaling time courses."""

import torch
from torch import Tensor, nn


class TimecourseClassifier(nn.Module):
    """Combine absolute signal levels with first-order temporal differences."""

    def __init__(
        self,
        *,
        input_length: int = 288,
        num_classes: int = 3,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if input_length < 16:
            raise ValueError("input_length must be at least 16.")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.input_length = input_length
        self.raw_branch = _feature_branch((7, 5, 3), dropout)
        self.derivative_branch = _feature_branch((9, 7, 5), dropout)
        feature_count = self._infer_feature_count()
        self.classifier = nn.Sequential(
            nn.Linear(feature_count, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(16),
            nn.Dropout(dropout),
            nn.Linear(16, num_classes),
        )

    def forward(self, timecourses: Tensor) -> Tensor:
        if timecourses.ndim != 3 or timecourses.shape[1] != 1:
            raise ValueError("timecourses must have shape (batch, 1, time).")
        if timecourses.shape[2] != self.input_length:
            raise ValueError(f"Expected {self.input_length} timepoints.")

        differences = torch.diff(timecourses, dim=2)
        features = torch.cat(
            (self.raw_branch(timecourses), self.derivative_branch(differences)),
            dim=1,
        )
        return self.classifier(features)

    def _infer_feature_count(self) -> int:
        previous_mode = self.training
        self.eval()
        with torch.no_grad():
            raw = self.raw_branch(torch.zeros(1, 1, self.input_length)).shape[1]
            diff = self.derivative_branch(torch.zeros(1, 1, self.input_length - 1)).shape[1]
        self.train(previous_mode)
        return raw + diff


def _feature_branch(kernel_sizes: tuple[int, int, int], dropout: float) -> nn.Sequential:
    channels = (1, 64, 32, 16)
    layers: list[nn.Module] = []
    for index, kernel_size in enumerate(kernel_sizes):
        layers.extend(
            [
                nn.Conv1d(channels[index], channels[index + 1], kernel_size, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(channels[index + 1]),
                nn.Dropout(dropout),
            ]
        )
    layers.append(nn.Flatten())
    return nn.Sequential(*layers)
