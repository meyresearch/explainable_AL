import numpy as np
import torch
import gpytorch
from torch.distributions import Normal

from .train_gp_model import train_gp_model as _train_gp_model_impl


class TanimotoKernel(gpytorch.kernels.Kernel):
    """Tanimoto kernel compatible with the GP regression utilities.

    Implements a numerically-stable Tanimoto similarity for
    fingerprint-like vectors.
    """

    def __init__(self, **kwargs):
        super(TanimotoKernel, self).__init__(**kwargs)

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            return torch.ones_like(x1[:, 0])
        else:
            x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
            x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
            x1_dot_x2 = torch.matmul(x1, x2.transpose(-1, -2))
            denominator = x1_norm + x2_norm.transpose(-1, -2) - x1_dot_x2
            return x1_dot_x2 / denominator.clamp(min=1e-9)


class GPRegressionModel(gpytorch.models.ExactGP):
    """GP regression model using the package's `TanimotoKernel` by default."""

    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = TanimotoKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x.add_jitter(1e-6))


# Re-export the richer train routine from train_gp_model.py
def train_gp_model(train_x, train_y, likelihood, model, epochs=50, lr=0.1, lr_decay=0.95):
    """Train a GP model (wrapper around the canonical implementation).

    See `train_gp_model.py` for details.
    """
    return _train_gp_model_impl(train_x, train_y, likelihood, model, epochs=epochs, lr=lr, lr_decay=lr_decay)


def ucb_selection(fingerprints, model, likelihood, batch_size, alpha, beta, already_selected_indices):
    """Vectorised UCB selection: return top `batch_size` indices by UCB score."""
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
    """Probability of Improvement selection (vectorised)."""
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pool_indices = list(set(range(len(fingerprints))) - set(already_selected_indices))
        pool_fingerprints = torch.tensor(np.array([np.array(fp) for fp in fingerprints])[pool_indices]).float()
        predictions = likelihood(model(pool_fingerprints))
        mean = predictions.mean
        std = predictions.stddev
        Z = (mean - current_best_y - xi) / (std + 1e-9)
        normal = Normal(torch.tensor(0.0), torch.tensor(1.0))
        pi_scores = normal.cdf(Z)
        best_indices = torch.argsort(pi_scores, descending=True)[:batch_size]
        return np.array(pool_indices)[best_indices.cpu().numpy()]


def ei_selection(fingerprints, model, likelihood, batch_size, already_selected_indices, current_best_y, xi=0.01):
    """Expected Improvement selection (vectorised)."""
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pool_indices = list(set(range(len(fingerprints))) - set(already_selected_indices))
        pool_fingerprints = torch.tensor(np.array([np.array(fp) for fp in fingerprints])[pool_indices]).float()
        predictions = likelihood(model(pool_fingerprints))
        mean = predictions.mean
        std = predictions.stddev
        Z = (mean - current_best_y - xi) / (std + 1e-9)
        normal = Normal(torch.tensor(0.0), torch.tensor(1.0))
        ei_scores = (mean - current_best_y - xi) * normal.cdf(Z) + std * torch.exp(normal.log_prob(Z))
        best_indices = torch.argsort(ei_scores, descending=True)[:batch_size]
        return np.array(pool_indices)[best_indices.cpu().numpy()]


def active_learning(original_df, fingerprints, epochs, lr, lr_decay, selection_protocol, ucb_alpha=1.0, ucb_beta=1.0):
    """Iterative active learning loop coordinating acquisition and model updates.

    Returns (cycle_results, already_selected_indices, all_predictions, gp_model, likelihood)
    """
    import pandas as pd
    from .metrics_plots import calculate_metrics

    selected_df = pd.DataFrame(columns=original_df.columns)
    top_2p_count = 0
    top_5p_count = 0
    already_selected_indices = []
    cycle_results = []
    all_predictions = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initial full-data tensors (not used directly for training here)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    train_x = torch.tensor(fingerprints).float().to(device)
    train_y = torch.tensor(original_df['affinity'].values).float().to(device)
    gp_model = GPRegressionModel(train_x, train_y, likelihood).to(device)

    for cycle, (method, batch_size) in enumerate(selection_protocol):
        if method == "random":
            available_indices = list(set(range(len(original_df))) - set(already_selected_indices))
            new_indices = np.random.choice(available_indices, size=batch_size, replace=False)
        elif method == "ucb":
            new_indices = ucb_selection(fingerprints, gp_model, likelihood, batch_size=batch_size, 
                                        alpha=ucb_alpha, beta=ucb_beta, already_selected_indices=already_selected_indices)
        elif method == "explore":
            new_indices = ucb_selection(fingerprints, gp_model, likelihood, batch_size=batch_size, 
                                        alpha=0, beta=1, already_selected_indices=already_selected_indices)
        elif method == "exploit":
            new_indices = ucb_selection(fingerprints, gp_model, likelihood, batch_size=batch_size, 
                                        alpha=1, beta=0, already_selected_indices=already_selected_indices)
        else:
            raise ValueError(f"Unknown selection method: {method}")

        new_selection = original_df.iloc[new_indices]
        selected_df = pd.concat([selected_df, new_selection])
        top_2p_count += new_selection.get('top_2p', 0).sum() if 'top_2p' in new_selection else 0
        top_5p_count += new_selection.get('top_5p', 0).sum() if 'top_5p' in new_selection else 0
        already_selected_indices.extend(list(new_indices))

        train_x = torch.tensor(np.array([np.array(fp) for fp in fingerprints])[selected_df.index]).float().to(device)
        train_y = torch.tensor(selected_df['affinity'].values).float().to(device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        gp_model = GPRegressionModel(train_x, train_y, likelihood).to(device)
        gp_model, likelihood, _ = _train_gp_model_impl(train_x, train_y, likelihood, gp_model, epochs, lr, lr_decay)

        gp_model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            all_x = torch.tensor(fingerprints).float().to(device)
            predictions = likelihood(gp_model(all_x)).mean.cpu().numpy()

        all_predictions.append(predictions)

        r2, spearman = calculate_metrics(gp_model, likelihood, all_x, torch.tensor(original_df['affinity'].values).float().to(device))
        rmse = np.sqrt(np.mean((original_df['affinity'] - predictions)**2))

        cycle_results.append({
            'cycle': cycle,
            'top_2p': top_2p_count,
            'top_5p': top_5p_count,
            'r2': r2,
            'spearman': spearman,
            'compounds_acquired': len(selected_df),
            'rmse': rmse,
            'method': method
        })

        print(f"Cycle {cycle} ({method}): R2={r2:.4f} Spearman={spearman:.4f} Acquired={len(selected_df)}")

    return cycle_results, already_selected_indices, all_predictions, gp_model, likelihood
