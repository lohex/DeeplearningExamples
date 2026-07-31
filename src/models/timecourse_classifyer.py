"""PyTorch model for classifying single-cell signaling time courses."""

import torch
from torch import Tensor, nn


class TimecourseClassifier(nn.Module):
    """Classify trajectories from their levels and first-order differences.

    The raw-signal branch captures absolute signaling levels and broad temporal
    patterns. The derivative branch emphasizes transitions that can be hidden
    by differences in baseline level between individual cells.
    """

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
            raise ValueError("dropout must be in the interval [0, 1).")

        self.input_length = input_length
        self.raw_branch = _make_feature_branch(
            kernel_sizes=(7, 5, 3),
            dropout=dropout,
        )
        self.derivative_branch = _make_feature_branch(
            kernel_sizes=(9, 7, 5),
            dropout=dropout,
        )

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
        """Return unnormalized class scores for a batch of trajectories."""
        if timecourses.ndim != 3:
            raise ValueError("timecourses must have shape (batch, channel, time).")
        if timecourses.shape[1] != 1:
            raise ValueError("timecourses must contain exactly one input channel.")
        if timecourses.shape[2] != self.input_length:
            raise ValueError(
                f"Expected {self.input_length} timepoints, "
                f"received {timecourses.shape[2]}."
            )

        differences = timecourses[:, :, 1:] - timecourses[:, :, :-1]
        raw_features = self.raw_branch(timecourses)
        derivative_features = self.derivative_branch(differences)
        combined_features = torch.cat(
            (raw_features, derivative_features),
            dim=1,
        )
        return self.classifier(combined_features)

    def _infer_feature_count(self) -> int:
        raw_example = torch.zeros(1, 1, self.input_length)
        derivative_example = torch.zeros(1, 1, self.input_length - 1)
        previous_training_mode = self.training
        self.eval()

        with torch.no_grad():
            raw_count = self.raw_branch(raw_example).shape[1]
            derivative_count = self.derivative_branch(derivative_example).shape[1]

        self.train(mode=previous_training_mode)
        return raw_count + derivative_count


def _make_feature_branch(
    *,
    kernel_sizes: tuple[int, int, int],
    dropout: float,
) -> nn.Sequential:
    channels = (1, 64, 32, 16)
    layers: list[nn.Module] = []

    for layer_index, kernel_size in enumerate(kernel_sizes):
        input_channels = channels[layer_index]
        output_channels = channels[layer_index + 1]
        layers.extend(
            (
                nn.Conv1d(
                    input_channels,
                    output_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=1,
                ),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(output_channels),
                nn.Dropout(dropout),
            )
        )

    layers.append(nn.Flatten())
    return nn.Sequential(*layers)
