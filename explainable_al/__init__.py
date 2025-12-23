"""Explainable Active Learning package

Public API exports for core functionality.
"""
from .active_learning_core import (
    TanimotoKernel,
    GPRegressionModel,
    train_gp_model,
    ucb_selection,
    pi_selection,
    ei_selection,
    run_active_learning_experiment,
)

from .train_gp_model import train_gp_model as train_gp_model_exact
from .gpregression import GPRegressionModel as GPRegressionModelExact
from .acquisition_function import ucb_selection as ucb_selection_v1
from .featuriser import smiles_to_ecfp8_df, get_maccs_from_smiles_list, load_chemberta_embeddings

__all__ = [
    "TanimotoKernel",
    "GPRegressionModel",
    "train_gp_model",
    "ucb_selection",
    "pi_selection",
    "ei_selection",
    "run_active_learning_experiment",
    "smiles_to_ecfp8_df",
    "get_maccs_from_smiles_list",
    "load_chemberta_embeddings",
]
