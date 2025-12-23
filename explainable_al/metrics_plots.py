from .utils import *


def calculate_metrics(model, likelihood, test_x, test_y):
    """Compute R^2 and Spearman correlation between model predictions and targets.

    Parameters
    ----------
    model : gpytorch.models.ExactGP
        Trained GP model.
    likelihood : gpytorch.likelihoods.Likelihood
        Likelihood wrapper used to obtain predictions.
    test_x : torch.Tensor
        Input features for evaluation.
    test_y : torch.Tensor
        Ground-truth target values.

    Returns
    -------
    (float, float)
        Tuple of (r2, spearman_correlation).
    """
    model.eval()
    likelihood.eval()
    device = test_x.device

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = likelihood(model(test_x)).mean

    r2 = r2_score(test_y.cpu().numpy(), predictions.cpu().numpy())
    spearman_corr, _ = spearmanr(test_y.cpu().numpy(), predictions.cpu().numpy())
    return r2, spearman_corr


def prepare_data_for_plot(results):
    """Convert experiment `results` into a tidy DataFrame for plotting.

    The function expects `results` to be a mapping from protocol -> runs where
    each run includes `cycle_counts` describing per-cycle statistics.
    """
    data_list = []
    for protocol, protocol_data in results.items():
        for run in protocol_data:
            cycle_counts = run['cycle_counts']
            dataset_size = run.get('dataset_size', len(cycle_counts))
            compounds_acquired = np.cumsum([cycle['batch_size'] for cycle in cycle_counts])
            
            for i, cycle in enumerate(cycle_counts):
                top_2p = cycle.get('top_2p', 0)
                top_5p = cycle.get('top_5p', 0)
                recall_2p = top_2p / (0.02 * dataset_size) if dataset_size > 0 else 0
                recall_5p = top_5p / (0.05 * dataset_size) if dataset_size > 0 else 0
                data_list.append({
                    'Protocol': protocol,
                    'Compounds acquired': compounds_acquired[i],
                    'Recall (2%)': recall_2p,
                    'Recall (5%)': recall_5p,
                    'Dataset': run.get('dataset_name', 'Unknown'),
                    'Model': run.get('model_name', 'Unknown'),
                    'Lower CI': max(0, recall_2p - 0.02 * recall_2p),
                    'Upper CI': min(1, recall_2p + 0.02 * recall_2p)
                })
    return pd.DataFrame(data_list)


def make_plot_recall(data: pd.DataFrame, y: str = "Recall (2%)"):
    """Plot recall curves grouped by protocol, dataset and model.

    Parameters
    ----------
    data : pandas.DataFrame
        Tidy dataframe produced by `prepare_data_for_plot`.
    y : str
        Column name to plot on the y-axis (default: 'Recall (2%)').
    """
    sns.set_style("white")
    font_sizes = {'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16, 'xtick.labelsize': 12, 'ytick.labelsize': 12}
    sns.set_context("talk", rc=font_sizes)

    DATASET_ORDER = sorted(data['Dataset'].unique())
    PROTOCOL_ORDER = sorted(data['Protocol'].unique())

    if 'Model' in data.columns:
        MODEL_ORDER = sorted(data['Model'].unique())
    else:
        MODEL_ORDER = ['Default']
        data['Model'] = 'Default'

    palette = sns.color_palette("Set2", len(PROTOCOL_ORDER))

    g = sns.relplot(data=data, 
                    x="Compounds acquired", 
                    y=y, 
                    hue="Protocol",
                    row="Model", 
                    col="Dataset", 
                    kind="line", 
                    height=3, 
                    aspect=1.2,
                    facet_kws={"sharey": False, "sharex": True},
                    hue_order=PROTOCOL_ORDER,
                    row_order=MODEL_ORDER,
                    col_order=DATASET_ORDER,
                    linewidth=2.5,
                    palette=palette
                    )
    g.set_titles("")
    for ax, title in zip(g.axes[0], DATASET_ORDER):
        ax.set_title(title)
    for ax, row_name in zip(g.axes[:,0], MODEL_ORDER):
        ax.annotate(row_name, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center', rotation=90)
    g.set_axis_labels("Compounds acquired", y)
    g.tight_layout()
    for ax in g.axes.flat:
        y_max = ax.get_ylim()[1]
        y_steps = [0, 0.25*y_max, 0.5*y_max, 0.75*y_max, y_max]
        for yv in y_steps:
            ax.axhline(yv, color='gray', linestyle='dashed', alpha=0.1)
        for x in [100, 200, 300]:
            ax.axvline(x, color='gray', linestyle='dashed', alpha=0.1)

    plt.subplots_adjust(top=0.92)
    plt.show()
