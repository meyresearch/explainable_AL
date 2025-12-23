"""Compatibility shim: re-export canonical API from `active_learning`.

This module used to contain the core primitive implementations. After
consolidation, the canonical implementations live in
`explainable_al.active_learning`. To preserve backwards compatibility,
this module re-exports those symbols and exposes the legacy experiment
runner from `explainable_al.experiments` as `run_active_learning_experiment`.
"""

from .active_learning import (
    TanimotoKernel,
    GPRegressionModel,
    train_gp_model,
    ucb_selection,
    pi_selection,
    ei_selection,
    active_learning,
)

from .experiments import run_experiment as run_active_learning_experiment

__all__ = [
    "TanimotoKernel",
    "GPRegressionModel",
    "train_gp_model",
    "ucb_selection",
    "pi_selection",
    "ei_selection",
    "active_learning",
    "run_active_learning_experiment",
]
