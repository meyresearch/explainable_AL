"""Thin compatibility wrapper to use the canonical package implementation.

This module exists for code that imports `apps.al_app.active_learning_core`.
It re-exports the implementations from `explainable_al.active_learning_core`.
"""
import torch
import gpytorch
from torch.distributions import Normal
import numpy as np


from explainable_al.active_learning_core import (
    TanimotoKernel,
    GPRegressionModel,
    train_gp_model,
    ucb_selection,
    pi_selection,
    ei_selection,
    run_active_learning_experiment,
)

__all__ = [
    "TanimotoKernel",
    "GPRegressionModel",
    "train_gp_model",
    "ucb_selection",
    "pi_selection",
    "ei_selection",
    "run_active_learning_experiment",
]


def ucb_selection(fingerprints, model, likelihood, batch_size, alpha, beta, already_selected_indices):
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pool_indices = list(set(range(len(fingerprints))) - set(already_selected_indices))
        pool_fingerprints = torch.tensor(np.array([np.array(fp) for fp in fingerprints])[pool_indices]).float()
        predictions = likelihood(model(pool_fingerprints))
        mean = predictions.mean
        std = predictions.stddev
        ucb_scores = alpha * mean + beta * std
        best_indices = torch.argsort(ucb_scores, descending=True)[:batch_size]
        return np.array(pool_indices)[best_indices.cpu().numpy()]


def pi_selection(fingerprints, model, likelihood, batch_size, already_selected_indices, current_best_y, xi=0.01):
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pool_indices = list(set(range(len(fingerprints))) - set(already_selected_indices))
        pool_fingerprints = torch.tensor(np.array([np.array(fp) for fp in fingerprints])[pool_indices]).float()
        predictions = likelihood(model(pool_fingerprints))
        mean = predictions.mean
        std = predictions.stddev

        # Calculate Z-score
        Z = (mean - current_best_y - xi) / (std + 1e-9) # Add small epsilon for numerical stability
        
        normal = Normal(torch.tensor(0.0), torch.tensor(1.0))
        pi_scores = normal.cdf(Z)
        
        best_indices = torch.argsort(pi_scores, descending=True)[:batch_size]
        return np.array(pool_indices)[best_indices.cpu().numpy()]


def ei_selection(fingerprints, model, likelihood, batch_size, already_selected_indices, current_best_y, xi=0.01):
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pool_indices = list(set(range(len(fingerprints))) - set(already_selected_indices))
        pool_fingerprints = torch.tensor(np.array([np.array(fp) for fp in fingerprints])[pool_indices]).float()
        predictions = likelihood(model(pool_fingerprints))
        mean = predictions.mean
        std = predictions.stddev

        # Calculate Z-score
        Z = (mean - current_best_y - xi) / (std + 1e-9) # Add small epsilon for numerical stability
        
        normal = Normal(torch.tensor(0.0), torch.tensor(1.0))
        ei_scores = (mean - current_best_y - xi) * normal.cdf(Z) + std * torch.exp(normal.log_prob(Z))
        
        best_indices = torch.argsort(ei_scores, descending=True)[:batch_size]
        return np.array(pool_indices)[best_indices.cpu().numpy()]


