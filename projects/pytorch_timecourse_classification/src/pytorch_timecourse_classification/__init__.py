"""PyTorch classification of single-cell time courses."""

from .experiments import create_model, fit_model, save_and_verify_experiment
from .models import (
    AttentionClassifier,
    CNNClassifier,
    CNNAttentionPoolingClassifier,
    ExtendedCNNClassifier,
    HierarchicalPatchTransformerClassifier,
)
from .training import EpochMetrics, TrainingConfig, TrainingHistory, train
from .tuning import (
    AblationSpec,
    OptunaScanConfig,
    add_paired_reference_deltas,
    crossed_ablation_specs,
    optuna_parameter_importances,
    run_ablations,
    run_optuna_scan,
    save_ablation_study,
    summarize_ablations,
)

__all__ = [
    "AttentionClassifier",
    "CNNClassifier",
    "CNNAttentionPoolingClassifier",
    "ExtendedCNNClassifier",
    "HierarchicalPatchTransformerClassifier",
    "TrainingConfig",
    "TrainingHistory",
    "EpochMetrics",
    "train",
    "create_model",
    "fit_model",
    "save_and_verify_experiment",
    "AblationSpec",
    "add_paired_reference_deltas",
    "OptunaScanConfig",
    "crossed_ablation_specs",
    "optuna_parameter_importances",
    "run_ablations",
    "run_optuna_scan",
    "save_ablation_study",
    "summarize_ablations",
]
