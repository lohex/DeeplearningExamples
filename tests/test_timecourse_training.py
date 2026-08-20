import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from pytorch_timecourse_classification.artifacts import load_model, save_experiment_artifacts
from pytorch_timecourse_classification.data import FoldData, Preprocessor
from pytorch_timecourse_classification.models.cnn import CNNClassifier
from pytorch_timecourse_classification.training import TrainingConfig, train
from pytorch_timecourse_classification.tuning import (
    AblationSpec,
    add_paired_reference_deltas,
    run_ablations,
    summarize_ablations,
)


def synthetic_fold(seed: int, sample_count: int = 12) -> FoldData:
    rng = np.random.default_rng(seed)
    targets = np.arange(sample_count) % 2
    trajectories = rng.normal(size=(sample_count, 16)).astype(np.float32)
    trajectories += targets[:, None] * 0.8
    return FoldData(
        trajectories=trajectories,
        doses=np.where(targets == 0, "0pM", "1pM"),
        years=np.full(sample_count, 2020),
        features=torch.from_numpy(trajectories[:, None, :]),
        targets=torch.tensor(targets, dtype=torch.long),
    )


class NaNClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(16, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features.flatten(1)) * float("nan")


class TimecourseTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train_fold = synthetic_fold(1)
        self.tune_fold = synthetic_fold(2, sample_count=8)
        self.preprocessor = Preprocessor(("0pM", "1pM"), 0.0, 1.0)
        self.config = TrainingConfig(
            epochs=2,
            batch_size=4,
            learning_rate=1e-3,
            weight_decay=0.0,
            scheduler_strategy="none",
            patience=2,
            report_every=2,
            class_balance_strategy="none",
            gradient_clip_norm=1.0,
        )
        self.model_config = {
            "channels": (2,),
            "kernel_sizes": (3,),
            "dilations": (1,),
            "pool_size": 2,
            "hidden_dim": 2,
            "dropout": 0.0,
            "residual": False,
        }

    def test_training_history_contains_complete_epoch_series(self) -> None:
        model = CNNClassifier(input_length=16, num_classes=2, **self.model_config)
        history = train(
            model,
            self.train_fold.features,
            self.train_fold.targets,
            self.tune_fold.features,
            self.tune_fold.targets,
            config=self.config,
            device="cpu",
        )
        for field in (
            "training_loss",
            "validation_loss",
            "training_accuracy",
            "validation_accuracy",
            "training_macro_f1",
            "validation_macro_f1",
            "learning_rate",
        ):
            values = np.asarray(getattr(history, field))
            self.assertEqual(len(values), 2)
            self.assertTrue(np.isfinite(values).all())

        with tempfile.TemporaryDirectory() as directory:
            paths = save_experiment_artifacts(
                Path(directory),
                model=model,
                history=history,
                preprocessor=self.preprocessor,
                training_targets=self.train_fold.targets,
                validation_targets=self.tune_fold.targets,
                device="cpu",
            )
            with np.load(paths.history) as saved_history:
                self.assertIn("training_macro_f1", saved_history)
                self.assertIn("validation_macro_f1", saved_history)
                self.assertIn("learning_rate", saved_history)
            restored, _ = load_model(paths.model, device="cpu")
            model.eval()
            with torch.no_grad():
                expected_logits = model(self.tune_fold.features)
                restored_logits = restored(self.tune_fold.features)
            torch.testing.assert_close(
                restored_logits, expected_logits, rtol=0.0, atol=0.0
            )

    def test_non_finite_metrics_raise_explicit_error(self) -> None:
        with self.assertRaisesRegex(FloatingPointError, "Non-finite"):
            train(
                NaNClassifier(),
                self.train_fold.features,
                self.train_fold.targets,
                self.tune_fold.features,
                self.tune_fold.targets,
                config=self.config,
                device="cpu",
            )

    def test_paired_deltas_and_summary(self) -> None:
        frame = pd.DataFrame(
            {
                "name": ["reference", "candidate", "reference", "candidate"],
                "random_seed": [42, 42, 43, 43],
                "tune_loss": [1.0, 0.9, 1.1, 1.0],
                "tune_accuracy": [0.5, 0.6, 0.4, 0.5],
                "tune_macro_f1": [0.40, 0.45, 0.50, 0.60],
            }
        )
        paired = add_paired_reference_deltas(frame, reference_name="reference")
        candidate_deltas = paired.loc[
            paired["name"] == "candidate", "tune_macro_f1_delta"
        ].to_numpy()
        np.testing.assert_allclose(candidate_deltas, [0.05, 0.10])
        summary = summarize_ablations(paired)
        self.assertAlmostEqual(summary.loc["candidate", "macro_f1_delta_mean"], 0.075)

    def test_run_ablations_cpu_smoke(self) -> None:
        specs = (
            AblationSpec(
                name="reference",
                model_config=self.model_config,
                training_config=self.config,
                random_seeds=(42,),
            ),
        )
        results, frame = run_ablations(
            specs,
            model_type=CNNClassifier,
            training_fold=self.train_fold,
            tuning_fold=self.tune_fold,
            preprocessor=self.preprocessor,
            device="cpu",
        )
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].failed)
        self.assertEqual(int(frame.loc[0, "epochs_run"]), 2)
        self.assertTrue(np.isfinite(frame.loc[0, "duration_seconds"]))


if __name__ == "__main__":
    unittest.main()
