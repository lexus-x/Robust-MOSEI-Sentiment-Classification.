"""Robust multimodal sentiment experiments on CMU-MOSEI."""

from .config import ExperimentConfig, available_experiments, make_experiment_config

__all__ = [
    "ExperimentConfig",
    "available_experiments",
    "make_experiment_config",
]
