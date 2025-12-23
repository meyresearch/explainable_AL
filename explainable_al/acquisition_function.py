from .utils import *


def ucb_selection(fingerprints, model, likelihood, batch_size=30, alpha=1.0, beta=1.0, already_selected_indices=[]):
    """Select a batch of indices using the Upper Confidence Bound (UCB) acquisition.

    Parameters
    ----------
    fingerprints : Sequence
        Iterable of feature vectors (e.g., numpy arrays) for the pool.
    model : torch.nn.Module
        Trained GP model that accepts a single input and returns a multitask distribution.
    likelihood : gpytorch.likelihoods.Likelihood
        Likelihood wrapper used to obtain predictive mean and stddev.
    batch_size : int
        Number of indices to return.
    alpha : float
        Weight on the predictive mean in the acquisition score.
    beta : float
        Weight on the predictive standard deviation in the acquisition score.
    already_selected_indices : list
        Indices already selected (these will be excluded from the pool).

    Returns
    -------
    list
        List of selected indices (length <= batch_size).
    """
    device = next(model.parameters()).device 
    all_indices = set(range(len(fingerprints)))
    remaining_indices = list(all_indices - set(already_selected_indices))
    
    ucb_scores = []
    
    model.eval()
    likelihood.eval()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for idx in remaining_indices:
            test_x = torch.tensor(fingerprints[idx]).float().unsqueeze(0).to(device)
            pred = model(test_x)
            mu = pred.mean.item()
            sigma = pred.stddev.item()
            ucb_score = alpha * mu + beta * sigma
            ucb_scores.append((idx, ucb_score))
    
    ucb_scores.sort(key=lambda x: x[1], reverse=True)
    selected_indices = [x[0] for x in ucb_scores[:batch_size]]
    
    return selected_indices
