"""Experiment runner (was ecfp.py) — contains run_experiment entrypoint.
"""
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from .featuriser import smiles_to_ecfp8_df
from .active_learning import active_learning


# Example selection_protocols (kept minimal; adjust as needed)
selection_protocols = {
     "random": [("random", 60)] + [("random", 30)] * 10,
     "ucb-alternate": [("random", 60)] + [("explore", 30) if i % 2 == 0 else ("exploit", 30) for i in range(10)],
     "ucb-sandwich": [("random", 60)] + [("explore", 30)] * 2 + [("exploit", 30)] * 6 + [("explore", 30)] * 2,
    "ucb-explore-heavy": [("random", 60)] + [("explore", 30)] * 7 + [("exploit", 30)] * 3,
    "ucb-exploit-heavy": [("random", 60)] + [("explore", 30)] * 3 + [("exploit", 30)] * 7,
     "ucb-gradual": [("random", 60)] + [("explore", 30)] * 3 + [("ucb", 30)] * 4 + [("exploit", 30)] * 3,
    "ucb-balanced": [("random", 60)] + [("ucb", 30)] * 10,
 }

def run_experiment(dataset_path, dataset_name):
    """Entrypoint to run experiments locally.

    Parameters
    ----------
    dataset_path : str
        Path to a CSV file containing at minimum a `SMILES` column and target values.
    dataset_name : str
        Short name used for reporting/outputs.

    Returns
    -------
    results : dict
        Mapping protocol_name -> per-cycle results.
    dataset_size : int
        Number of rows in the dataset.
    """
    original_df = pd.read_csv(dataset_path)
    fingerprints = smiles_to_ecfp8_df(original_df, 'SMILES')
    epochs = 150
    lr = 0.01
    lr_decay = 0.95

    results = {}
    seed = 42
    for protocol_name, protocol in selection_protocols.items():
        np.random.seed(seed)
        torch.manual_seed(seed)
        cycle_results, selected_indices, all_predictions, final_model, final_likelihood = active_learning(
            original_df, fingerprints, epochs=epochs, lr=lr, lr_decay=lr_decay,
            selection_protocol=protocol)
        results[protocol_name] = cycle_results

    return results, len(original_df)
