import unittest

import numpy as np
import pandas as pd
import torch

from pytorch_timecourse_classification.data import FoldData, subset_fold
from pytorch_timecourse_classification.experiments import trainable_parameter_count
from pytorch_timecourse_classification.models.cnn import CNNClassifier, scaled_cnn_config
from pytorch_timecourse_classification.scaling import (
    nested_stratified_indices,
    select_best_learning_rate,
)


def synthetic_fold(sample_count: int = 24) -> FoldData:
    targets = np.arange(sample_count) % 3
    trajectories = np.arange(sample_count * 16, dtype=np.float32).reshape(sample_count, 16)
    return FoldData(
        trajectories=trajectories,
        doses=np.asarray([f"{target}pM" for target in targets], dtype=object),
        years=np.full(sample_count, 2020),
        features=torch.from_numpy(trajectories[:, None, :]),
        targets=torch.tensor(targets, dtype=torch.long),
    )


class TimecourseScalingTests(unittest.TestCase):
    def test_scaled_cnn_parameter_ladder(self) -> None:
        expected = {
            0.25: 4326,
            0.5: 16582,
            1.0: 64902,
            2.0: 256774,
            4.0: 1021446,
        }
        for width, parameter_count in expected.items():
            model = CNNClassifier(
                input_length=289,
                num_classes=6,
                **scaled_cnn_config(width),
            )
            self.assertEqual(trainable_parameter_count(model), parameter_count)

    def test_subset_fold_keeps_fields_aligned(self) -> None:
        fold = synthetic_fold()
        indices = np.asarray([1, 5, 9, 13], dtype=np.int64)
        subset = subset_fold(fold, indices)
        np.testing.assert_array_equal(subset.trajectories, fold.trajectories[indices])
        np.testing.assert_array_equal(subset.doses, fold.doses[indices])
        torch.testing.assert_close(subset.features, fold.features[indices])
        torch.testing.assert_close(subset.targets, fold.targets[indices])

    def test_nested_stratified_indices_are_nested(self) -> None:
        fold = synthetic_fold(60)
        subsets = nested_stratified_indices(
            fold.targets,
            (0.125, 0.25, 0.5, 1.0),
            random_seed=7,
        )
        previous: set[int] = set()
        for indices in subsets.values():
            current = set(indices.tolist())
            self.assertTrue(previous.issubset(current))
            self.assertEqual(set(fold.targets[indices].tolist()), {0, 1, 2})
            previous = current
        self.assertEqual(len(subsets[1.0]), len(fold.targets))

    def test_best_learning_rate_is_selected_per_n_p_cell(self) -> None:
        runs = pd.DataFrame(
            {
                "data_fraction": [0.5, 0.5, 0.5, 1.0, 1.0, 1.0],
                "n_samples": [50, 50, 50, 100, 100, 100],
                "width_multiplier": [1.0] * 6,
                "parameter_count": [1000] * 6,
                "learning_rate": [3e-4, 1e-3, 3e-3] * 2,
                "tune_loss": [0.8, 0.6, 0.7, 0.5, 0.55, 0.65],
                "failed": [False] * 6,
            }
        )
        selected = select_best_learning_rate(runs)
        self.assertEqual(selected["learning_rate"].tolist(), [1e-3, 3e-4])


if __name__ == "__main__":
    unittest.main()
