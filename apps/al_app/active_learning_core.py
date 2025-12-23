"""Thin compatibility wrapper to use the canonical package implementation.

This module exists for code that imports `apps.al_app.active_learning_core`.
It re-exports the implementations from `explainable_al.active_learning_core`.
"""

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


def run_active_learning_experiment(original_df, fingerprints, kernel, selection_protocol, protocol_name, y_column):
    from frontend.utils import calculate_metrics
    import pandas as pd

    total_top_2p = original_df['top_2p'].sum()
    total_top_5p = original_df['top_5p'].sum()

    selected_df = pd.DataFrame(columns=original_df.columns)
    already_selected_indices = []
    cycle_results = []
    top_2p_count = 0
    top_5p_count = 0
    fp_array = np.array([np.array(fp) for fp in fingerprints])

    for i, protocol_step in enumerate(selection_protocol):
        method, *params = protocol_step
        batch_size = params[0]

        if method == "random":
            available_indices = list(set(range(len(original_df))) - set(already_selected_indices))
            new_indices = np.random.choice(available_indices, size=batch_size, replace=False)
        else:
            train_x = torch.tensor(fp_array[already_selected_indices]).float()
            train_y = torch.tensor(original_df.iloc[already_selected_indices][y_column].values).float()
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = GPRegressionModel(train_x, train_y, likelihood, kernel)
            model, likelihood = train_gp_model(train_x, train_y, likelihood, model)

            if method == "explore" or method == "exploit":
                alpha, beta = (0, 1) if method == "explore" else (1, 0)
                new_indices = ucb_selection(fingerprints, model, likelihood, batch_size, alpha, beta, already_selected_indices)
            elif method == "PI":
                xi_value = params[1] if len(params) > 1 else 0.01
                current_best_y = train_y.max()
                new_indices = pi_selection(fingerprints, model, likelihood, batch_size, already_selected_indices, current_best_y, xi=xi_value)
            elif method == "EI":
                xi_value = params[1] if len(params) > 1 else 0.01
                current_best_y = train_y.max()
                new_indices = ei_selection(fingerprints, model, likelihood, batch_size, already_selected_indices, current_best_y, xi=xi_value)

        new_selection = original_df.iloc[new_indices]
        selected_df = pd.concat([selected_df, new_selection])
        already_selected_indices.extend(new_indices)
        top_2p_count += new_selection['top_2p'].sum()
        top_5p_count += new_selection['top_5p'].sum()

        final_train_x = torch.tensor(fp_array[already_selected_indices]).float()
        final_train_y = torch.tensor(original_df.iloc[already_selected_indices][y_column].values).float()
        final_likelihood = gpytorch.likelihoods.GaussianLikelihood()
        final_model = GPRegressionModel(final_train_x, final_train_y, final_likelihood, kernel)
        final_model, final_likelihood = train_gp_model(final_train_x, final_train_y, final_likelihood, final_model)
        test_x = torch.tensor(fp_array).float()
        test_y = torch.tensor(original_df[y_column].values).float()
        r2, spearman = calculate_metrics(final_model, final_likelihood, test_x, test_y)

        cycle_results.append({
            'Protocol': protocol_name,
            'Cycle': i + 1,
            'Method': method,
            'Compounds acquired': len(selected_df),
            'R2': r2,
            'Spearman': spearman,
            'Recall (2%)': top_2p_count / total_top_2p if total_top_2p > 0 else 0,
            'Recall (5%)': top_5p_count / total_top_5p if total_top_5p > 0 else 0,
        })
        print(f"Cycle {i+1}/{len(selection_protocol)} complete for protocol: {protocol_name}")

    return pd.DataFrame(cycle_results)
