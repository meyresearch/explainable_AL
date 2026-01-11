import streamlit as st
import pandas as pd
import numpy as np
import torch
import gpytorch
import traceback

from explainable_al.active_learning_core import (
    TanimotoKernel,
    GPRegressionModel,
    train_gp_model,
    ucb_selection,
    pi_selection,
    ei_selection,
)
from explainable_al import metrics_plots
from explainable_al import active_learning_core as _alc
from explainable_al import utils as _alutils
from explainable_al.active_learning_core import run_active_learning_experiment
from explainable_al.metrics_plots import make_plot_recall
from explainable_al.active_learning_core import TanimotoKernel as TanimotoKernel
from explainable_al.active_learning_core import GPRegressionModel as GPRegressionModel
from explainable_al.active_learning_core import train_gp_model as train_gp_model
from explainable_al.active_learning_core import ucb_selection as ucb_selection
from explainable_al.active_learning_core import pi_selection as pi_selection
from explainable_al.active_learning_core import ei_selection as ei_selection
from explainable_al import utils as utils_module

# --- Pre-defined Protocols --- #
selection_protocols = {
    "ucb-explore-heavy": [("random", 60)] + [("explore", 30)] * 7 + [("exploit", 30)] * 3,
    "ucb-exploit-heavy": [("random", 60)] + [("explore", 30)] * 3 + [("exploit", 30)] * 7,
}

def run_active_learning_cycle(original_df, fingerprints, kernel, selection_protocol, protocol_name, y_column):
    """
    Runs a full active learning cycle for a given protocol.

    Args:
        original_df: The original DataFrame.
        fingerprints: List of fingerprints.
        kernel: The kernel to use.
        selection_protocol: The selection protocol.
        protocol_name (str): The name of the protocol.
        y_column (str): The name of the target column.

    Returns:
        pd.DataFrame: A DataFrame containing the results of the active learning cycle.
    """
    st.subheader(f"Running Protocol: {protocol_name}")
    progress_bar = st.progress(0)
    results_placeholder = st.empty()
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

        # For metrics, retrain on all selected data
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
        results_df_current_protocol = pd.DataFrame(cycle_results)
        results_placeholder.dataframe(results_df_current_protocol)
        progress_bar.progress((i + 1) / len(selection_protocol))

    st.success(f"Active learning process finished for {protocol_name}.")
    return pd.DataFrame(cycle_results)

def main():
    """
    Main function for the Streamlit application.
    """
    st.title('Active Learning for Drug Discovery')

    st.header('1. Upload Your Dataset')
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
        r2, spearman = utils_module.calculate_metrics(final_model, final_likelihood, test_x, test_y)
    if 'df' in st.session_state:
        st.dataframe(st.session_state.df.head())
        df = st.session_state.df
        st.header('2. Define Task')
        x_column = st.selectbox('Select the feature column (X) for SMILES', df.columns, key='x_col')
        y_column = st.selectbox('Select the target column (Y)', df.columns, key='y_col')

        st.header('3. Molecular Representation')
        representation = st.selectbox('Choose a molecular representation', ['ECFP Fingerprints', 'MACCS Keys', 'ChemBERTa Embeddings'], key='rep_type')

        chemberta_file = None
        if representation == 'ChemBERTa Embeddings':
            chemberta_file = st.file_uploader("Upload ChemBERTa Embeddings (.npz)", type="npz", key="chemberta_uploader")

        if st.button('Generate Representations'):
            with st.spinner("Generating representations..."):
                smiles_list = df[x_column].tolist()
                if representation == 'ECFP Fingerprints':
                    st.session_state.representations = utils_module.get_ecfp_fingerprints(smiles_list)
                elif representation == 'MACCS Keys':
                    st.session_state.representations = utils_module.get_maccs_keys(smiles_list)
                elif representation == 'ChemBERTa Embeddings':
                    if chemberta_file is not None:
                        st.session_state.representations = utils_module.get_chemberta_embeddings(chemberta_file)
                    else:
                        st.error("Please upload a ChemBERTa Embeddings .npz file.")
                        st.session_state.representations = None
                if st.session_state.representations is not None:
                    st.success("Representations generated successfully!")

        if 'representations' in st.session_state:
            st.header('4. Surrogate Model')
            kernel_name = st.selectbox('Choose a kernel', ['Tanimoto', 'RBF', 'Matern', 'Linear', 'Rational Quadratic'], key='kernel_name')
            if kernel_name == 'Tanimoto':
                st.session_state.gp_kernel = TanimotoKernel()
            elif kernel_name == 'RBF':
                st.session_state.gp_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            elif kernel_name == 'Matern':
                st.session_state.gp_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())
            elif kernel_name == 'Linear':
                st.session_state.gp_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
            elif kernel_name == 'Rational Quadratic':
                st.session_state.gp_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())
            st.success(f"{kernel_name} kernel selected.")

            st.header('5. Acquisition Protocol')
            
            pre_defined_protocol_names = list(selection_protocols.keys())
            selected_pre_defined_protocols = st.multiselect('Choose pre-defined protocols to compare', pre_defined_protocol_names, default=pre_defined_protocol_names[0] if pre_defined_protocol_names else [])

            st.subheader("Custom Protocol Configuration")
            use_custom_protocol = st.checkbox("Enable Custom Protocol", key="enable_custom_protocol")

            custom_selection_protocol = None
            custom_protocol_name = "Custom"

            if use_custom_protocol:
                initial_set_size = st.number_input("Size of initial random set", min_value=1, value=60, step=10, key='custom_initial')
                num_cycles = st.number_input("Number of active learning cycles", min_value=1, value=10, step=1, key='custom_cycles')
                batch_size = st.number_input("Number of compounds to select per cycle", min_value=1, value=30, step=5, key='custom_batch')
                
                acquisition_method = st.selectbox("Acquisition Method", ["UCB", "PI", "EI"], key="acquisition_method")

                if acquisition_method == "UCB":
                    explore_cycles = st.slider("Exploration Cycles", min_value=0, max_value=num_cycles, value=int(num_cycles * 0.7), step=1, key='custom_explore')
                    exploit_cycles = num_cycles - explore_cycles
                    st.write(f"{explore_cycles} cycles of exploration, {exploit_cycles} cycles of exploitation.")
                    custom_selection_protocol = [("random", initial_set_size)] + [("explore", batch_size)] * explore_cycles + [("exploit", batch_size)] * exploit_cycles
                elif acquisition_method in ["PI", "EI"]:
                    xi_value = st.number_input("xi value for PI/EI", min_value=0.0, value=0.01, step=0.01, key='xi_value')
                    custom_selection_protocol = [("random", initial_set_size)] + [(acquisition_method, batch_size, xi_value)] * num_cycles

            if st.button('Start Active Learning'):
                all_results_df = pd.DataFrame()

                protocols_to_run = []
                if selected_pre_defined_protocols:
                    for p_name in selected_pre_defined_protocols:
                        protocols_to_run.append({"name": p_name, "protocol": selection_protocols[p_name]})
                
                if use_custom_protocol and custom_selection_protocol:
                    protocols_to_run.append({"name": custom_protocol_name, "protocol": custom_selection_protocol})

                if not protocols_to_run:
                    st.error("No protocols selected or configured. Please select at least one pre-defined protocol or enable and configure a custom protocol.")
                    st.stop()

                original_df = st.session_state.df.copy()
                if 'top_2p' not in original_df.columns:
                    original_df['top_2p'] = (original_df[y_column] >= original_df[y_column].quantile(0.98))
                if 'top_5p' not in original_df.columns:
                    original_df['top_5p'] = (original_df[y_column] >= original_df[y_column].quantile(0.95))
                
                fingerprints = st.session_state.representations
                kernel = st.session_state.gp_kernel

                for current_protocol_info in protocols_to_run:
                    try:
                        results_df = run_active_learning_cycle(
                            original_df=original_df,
                            fingerprints=fingerprints,
                            kernel=kernel,
                            selection_protocol=current_protocol_info["protocol"],
                            protocol_name=current_protocol_info["name"],
                            y_column=y_column
                        )
                        all_results_df = pd.concat([all_results_df, results_df], ignore_index=True)
                    except Exception as e:
                        st.error(f"An error occurred during the active learning loop for {current_protocol_info['name']}: {e}")
                        st.code(traceback.format_exc())
                
                st.header("Final Results Comparison")
                st.dataframe(all_results_df)
                make_plot_recall(all_results_df, y='Recall (2%)')
                make_plot_recall(all_results_df, y='Recall (5%)')

if __name__ == "__main__":
    main()
