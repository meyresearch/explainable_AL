"""
SHAP analysis application for active learning in molecular property prediction.
"""

# Standard library imports
import os
import pickle
import zipfile
import io
import base64
import tempfile
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
import time
import warnings

# Third-party library imports
import streamlit as st
import pandas as pd
import numpy as np
import torch
import gpytorch
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.gridspec import GridSpec
from PIL import Image
from scipy.stats import gaussian_kde
from scipy import stats
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm

# Use canonical featurisers from the package to avoid duplication
from explainable_al.featuriser import (
    smiles_to_ecfp8_df,
    get_maccs_from_smiles_list,
    smiles_to_chemberta as _smiles_to_chemberta,
    load_chemberta_embeddings,
)

# Chemistry-specific imports
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, Draw, Descriptors, rdDepictor, rdMolDescriptors, MACCSkeys, rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem import rdRGroupDecomposition, rdMMPA
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

# Try to import optional packages
try:
    import umap.umap_ as umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Set random seeds and warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ==============================================================================
# 2. APPLICATION CONFIGURATION
# ==============================================================================
PAGE_TITLE = "Active Learning Analysis Platform"
PAGE_ICON = "🧬"
FINGERPRINT_RADIUS = 4
FINGERPRINT_BITS = 4096
AURA_HIGHLIGHT_COLOR = (1.0, 0.4, 0.8, 0.25)
AURA_HIGHLIGHT_RADIUS = 0.65

CUSTOM_APP_CSS = """
<style>
    .main-header {font-size: 3rem; color: #1f77b4; text-align: center; margin-bottom: 2rem;}
    .sub-header {font-size: 1.5rem; color: #ff7f0e; margin-top: 2rem; margin-bottom: 1rem; border-bottom: 2px solid #ff7f0e;}
    .info-box {background-color: #d1ecf1; border: 1px solid #bee5eb; border-radius: 0.5rem; padding: 1rem; margin: 1rem 0;}
    .stPlotlyChart {min-height: 550px;}
    .metric-card {background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 0.5rem; padding: 1rem; margin: 0.5rem 0;}
</style>
"""

# ==============================================================================
# 3. SHARED UTILITY FUNCTIONS
# ==============================================================================

# Define consistent colors for datasets and fingerprints
DATASET_COLORS = {
    'TYK2': '#1f77b4',  # Blue
    'USP7': '#ff7f0e',  # Orange
    'D2R':  '#2ca02c',  # Green
    'MPRO': '#d62728'   # Red
}

FP_COLORS = {
    'ECFP':      '#5D69B1',  # Deep Blue
    'MACCS':     '#52BCA3',  # Teal
    'ChemBERTa': "#E506DA"   # Magenta
}

# Define font sizes for readability
FONT_SIZES = {
    'title': 20,
    'label': 18,
    'tick': 16,
    'legend': 16,
    'annotation': 14
}

def apply_plot_style(ax):
    """Applies a consistent 'half-frame' style to a Matplotlib axis."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES['tick'])
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)

def create_molecule_aura_image(molecule, highlight_atoms, highlight_bonds):
    """Create molecule image with highlighted atoms and bonds."""
    if not molecule: return None
    drawer = rdMolDraw2D.MolDraw2DCairo(400, 300)
    drawer.drawOptions().padding = 0.1
    atom_highlight_map = {idx: [AURA_HIGHLIGHT_COLOR] for idx in highlight_atoms}
    bond_highlight_map = {idx: [AURA_HIGHLIGHT_COLOR] for idx in highlight_bonds}
    atom_radii_map = {idx: AURA_HIGHLIGHT_RADIUS for idx in highlight_atoms}
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule, highlightAtoms=atom_highlight_map, 
                                      highlightBonds=bond_highlight_map, highlightAtomRadii=atom_radii_map)
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText()))

def set_plot_style(plot_config):
    """Configure matplotlib plot style."""
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': plot_config['font_size'],
        'axes.labelsize': plot_config['label_size'],
        'axes.titlesize': plot_config['title_size'],
        'xtick.labelsize': plot_config['tick_size'],
        'ytick.labelsize': plot_config['tick_size'],
        'figure.dpi': plot_config['dpi'],
        'savefig.dpi': plot_config['dpi']
    })

def smiles_to_ecfp8(df, smiles_col):
    """Wrapper: compute ECFP bit-vectors using `explainable_al.featuriser.smiles_to_ecfp8_df`.

    Returns a NumPy array of shape (N, nBits) with integer bit values.
    """
    return smiles_to_ecfp8_df(df, smiles_col, radius=FINGERPRINT_RADIUS, nBits=FINGERPRINT_BITS)


def smiles_to_ecfp(smiles_df, radius=4, nBits=4096):
    """Alias for `smiles_to_ecfp8` kept for compatibility."""
    return smiles_to_ecfp8(smiles_df, 'SMILES')


def smiles_to_maccs(smiles_df):
    """Compute MACCS keys via the canonical featuriser.

    Returns a NumPy array of shape (N, 167).
    """
    return get_maccs_from_smiles_list(smiles_df['SMILES'].tolist())


def smiles_to_chemberta(smiles_df, batch_size=32):
    """Compute ChemBERTa embeddings using the package featuriser.

    Delegates to `explainable_al.featuriser.smiles_to_chemberta`.
    """
    return _smiles_to_chemberta(smiles_df, batch_size=batch_size)

def calculate_similarity_matrix_safe(fingerprints, metric='jaccard'):
    """Calculate similarity matrix safely."""
    from scipy.spatial.distance import pdist
    if len(fingerprints) < 2: return np.array([])
    is_zero = np.all(fingerprints == 0, axis=1)
    valid_fps = fingerprints[~is_zero]
    if len(valid_fps) < 2: return np.array([])
    
    dist = pdist(valid_fps, metric=metric)
    similarities = 1 - dist
    similarities = similarities[~np.isnan(similarities)]
    return similarities

# (rest of large SHAP app code omitted for brevity in this copy)
