from .utils import *
from .acquisition_function import ucb_selection
from .train_gp_model import train_gp_model
from .gpregression import GPRegressionModel
from .metrics_plots import calculate_metrics
def active_learning(original_df, fingerprints, epochs, lr, lr_decay, selection_protocol, ucb_alpha=1.0, ucb_beta=1.0, ucb_selection=ucb_selection):
    """Iterative active learning loop coordinating acquisition and model updates.

    Parameters
    ----------
    original_df : pandas.DataFrame
        Dataset containing SMILES/affinity and evaluation columns (top_2p/top_5p).
    fingerprints : Sequence
        Precomputed feature vectors aligning with rows in `original_df`.
    epochs : int
        Number of training epochs passed to the GP training routine.
    lr : float
        Learning rate for GP training.
    lr_decay : float
        Learning rate decay factor for the scheduler.
    selection_protocol : list
        Ordered list of (method, batch_size) tuples describing acquisition steps.
    ucb_alpha, ucb_beta : float
        Weights for mean and uncertainty when using UCB.

    Returns
    -------
    tuple
        (cycle_results, already_selected_indices, all_predictions, gp_model, likelihood)
        where `cycle_results` is a list of per-cycle metrics.
    """
    selected_df = pd.DataFrame(columns=original_df.columns)
    top_2p_count = 0
    top_5p_count = 0
    already_selected_indices = []
    cycle_results = []
    all_predictions = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        top_2p_count += new_selection['top_2p'].sum()
        top_5p_count += new_selection['top_5p'].sum()
        already_selected_indices.extend(new_indices)

        train_x = torch.tensor(fingerprints[selected_df.index]).float().to(device)
        train_y = torch.tensor(selected_df['affinity'].values).float().to(device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        gp_model = GPRegressionModel(train_x, train_y, likelihood).to(device)
        gp_model, likelihood, _ = train_gp_model(train_x, train_y, likelihood, gp_model, epochs, lr, lr_decay)

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

        print(f"Cycle {cycle} ({method}):")
        print(f"  Compounds selected: {len(selected_df)}")
        print(f"  Top 2% compounds selected: {top_2p_count}")
        print(f"  Top 5% compounds selected: {top_5p_count}")
        print(f"  R2: {r2:.4f}")
        print(f"  Spearman correlation: {spearman:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  Mean affinity of selected compounds: {selected_df['affinity'].mean():.2f}")
        print(f"  Mean affinity of entire dataset: {original_df['affinity'].mean():.2f}")

    return cycle_results, already_selected_indices, all_predictions, gp_model, likelihood
