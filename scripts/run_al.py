#!/usr/bin/env python3
"""
CLI runner for active learning experiments using the explainable_al package.
"""
import argparse
import os
import numpy as np
import pandas as pd

from explainable_al.featuriser import (
    smiles_to_ecfp8_df,
    get_maccs_from_smiles_list,
    load_chemberta_embeddings,
)
from explainable_al import active_learning as activelearning


def build_selection_protocol(protocol_name, initial_size, batch_size, cycles):
    protocol_steps = []
    protocol_steps.append(("random", initial_size))
    for _ in range(cycles):
        protocol_steps.append((protocol_name, batch_size))
    return protocol_steps


def main():
    p = argparse.ArgumentParser(description='Run active learning experiment (explainable_al)')
    p.add_argument('--dataset', required=True, help='Path to dataset CSV (must include SMILES and affinity columns)')
    p.add_argument('--representation', choices=['ecfp','maccs','chemberta'], default='ecfp')
    p.add_argument('--protocol', default='ucb', help='Selection method for cycles (ucb, explore, exploit, random)')
    p.add_argument('--initial-size', type=int, default=60)
    p.add_argument('--batch-size', type=int, default=30)
    p.add_argument('--cycles', type=int, default=10)
    p.add_argument('--output', required=True, help='Output directory (will be created)')
    args = p.parse_args()

    df = pd.read_csv(args.dataset)
    os.makedirs(args.output, exist_ok=True)

    # Compute or load features
    if args.representation == 'ecfp':
        fingerprints = smiles_to_ecfp8_df(df, 'SMILES')
    elif args.representation == 'maccs':
        fingerprints = get_maccs_from_smiles_list(df['SMILES'].tolist())
    else:  # chemberta
        candidate = os.path.splitext(args.dataset)[0] + '_chemberta.npz'
        if os.path.exists(candidate):
            fingerprints = load_chemberta_embeddings(candidate)
        else:
            raise FileNotFoundError(f'ChemBERTa embeddings not found: {candidate}\nCompute them with explainable_al.featuriser.smiles_to_chemberta or provide a .npz file')

    selection_protocol = build_selection_protocol(args.protocol, args.initial_size, args.batch_size, args.cycles)

    epochs = 150
    lr = 0.01
    lr_decay = 0.95

    cycle_results, selected_indices, all_predictions, final_model, final_likelihood = activelearning.active_learning(
        df, fingerprints, epochs=epochs, lr=lr, lr_decay=lr_decay, selection_protocol=selection_protocol
    )

    results_df = pd.DataFrame(cycle_results)
    csv_out = os.path.join(args.output, 'cycle_results.csv')
    results_df.to_csv(csv_out, index=False)

    npz_out = os.path.join(args.output, 'predictions.npz')
    np.savez_compressed(npz_out, predictions=np.array(all_predictions))

    print('Saved cycle results to', csv_out)
    print('Saved predictions to', npz_out)


if __name__ == '__main__':
    main()

