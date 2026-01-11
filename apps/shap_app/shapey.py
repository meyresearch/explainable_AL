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



#==============================================================================
# UNIFIED DRUG DISCOVERY ANALYSIS PLATFORM
# Combines SHAP-guided analysis with protocol performance visualization
# 
# Features:
# 1. SHAP-guided drug discovery:
#    - Chemical fragment mapping
#    - Feature evolution analysis
#    - Molecular design templates
#    - SAR analysis
# 
# 2. Protocol performance analysis:
#    - Performance comparisons
#    - Ridge plots
#    - Affinity predictions
#    - Chemical space visualization
# 
# 3. Publication-quality visualizations:
#    - Similarity distributions
#    - Property analysis
#    - Performance summaries
#    - Comprehensive heatmaps
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
    """Convert SMILES to ECFP fingerprints."""
    fingerprints = []
    for smiles in df[smiles_col]:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, FINGERPRINT_RADIUS, nBits=FINGERPRINT_BITS)
            fingerprints.append(fp)
        else:
            fingerprints.append(None)
    return np.array(fingerprints)

def smiles_to_ecfp(smiles_df, radius=4, nBits=4096):
    """Generate ECFP fingerprints using rdkit."""
    from rdkit.Chem import rdFingerprintGenerator
    from tqdm import tqdm
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    fingerprints = [mfpgen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else np.zeros(nBits) 
                   for s in tqdm(smiles_df['SMILES'], desc="ECFP")]
    return np.stack(fingerprints)

def smiles_to_maccs(smiles_df):
    """Generate MACCS fingerprints."""
    from rdkit.Chem import MACCSkeys
    from tqdm import tqdm
    fingerprints = [np.array(MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(s))) if Chem.MolFromSmiles(s) else np.zeros(167) 
                   for s in tqdm(smiles_df['SMILES'], desc="MACCS")]
    return np.stack(fingerprints)

def smiles_to_chemberta(smiles_df, batch_size=32):
    """Generate ChemBERTa embeddings."""
    from transformers import AutoTokenizer, AutoModel
    from tqdm import tqdm
    
    model_name = "DeepChem/ChemBERTa-77M-MTR"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    
    smiles_list = smiles_df['SMILES'].tolist()
    all_embeddings = []
    for i in tqdm(range(0, len(smiles_list), batch_size), desc="ChemBERTa"):
        batch = smiles_list[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings)

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

# ==============================================================================
# 4. CORE ANALYSIS CLASSES
# ==============================================================================
class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    
    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

class ChemicalFragmentMapper:
    def __init__(self, analysis_results, dataset_df, fingerprint_func):
        self.analysis_results = analysis_results
        self.dataset_df = dataset_df
        self.fingerprint_func = fingerprint_func
        self.fp_radius = FINGERPRINT_RADIUS
        self.fp_bits = FINGERPRINT_BITS
        self.fingerprints = self.fingerprint_func(self.dataset_df, 'SMILES')

    def extract_fragments_for_features(self, target, protocol, top_n, max_mols):
        df = self.analysis_results['feature_evolution']
        target_df = self.dataset_df[self.dataset_df['Target'].str.strip().str.upper() == target.upper()].copy().reset_index(drop=True)
        imp = df.groupby('feature_index')['importance'].mean()
        top_feats = imp.nlargest(top_n)
        frag_data = {}
        
        for rank, (idx, avg_imp) in enumerate(top_feats.items(), 1):
            mols_with_feat = [
                {'mol_idx': i, 'smiles': r['SMILES'], 'affinity': r['affinity']}
                for i, r in target_df.iterrows()
                if i < self.fingerprints.shape[0] and idx < self.fingerprints.shape[1] and self.fingerprints[i, idx] == 1
            ]
            if not mols_with_feat: continue
            mols_with_feat.sort(key=lambda x: x['affinity'], reverse=True)
            feat_frags = self._extract_substructures(idx, mols_with_feat[:max_mols])
            frag_data[idx] = {
                'feature_index': idx, 
                'average_importance': avg_imp, 
                'rank': rank, 
                'molecules_with_feature': mols_with_feat, 
                'fragments': feat_frags
            }
        return frag_data

    def _extract_substructures(self, feat_idx, mols):
        """Extract substructures with progress indicators."""
        fragments_info = {'common_smiles': [], 'molecule_highlights': []}
        
        progress_container = st.empty()
        status_container = st.empty()
        
        progress_container.info(f"🔍 Processing feature {feat_idx} with {len(mols)} molecules...")
        
        successful_extractions = 0
        highlight_only_count = 0
        
        for i, mol_info in enumerate(mols):
            progress = (i + 1) / len(mols)
            status_container.progress(progress, text=f"Analyzing molecule {i+1}/{len(mols)} (Affinity: {mol_info['affinity']:.2f})")
            
            mol = Chem.MolFromSmiles(mol_info['smiles'])
            if not mol: continue
            rdDepictor.Compute2DCoords(mol)
            
            bit_info = {}
            AllChem.GetMorganFingerprintAsBitVect(mol, radius=self.fp_radius, nBits=self.fp_bits, bitInfo=bit_info)
            
            if feat_idx in bit_info:
                fragment_found_for_mol = False
                for atom_idx, radius in bit_info[feat_idx]:
                    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
                    if not env: continue
                    
                    bonds = list(env)
                    atoms = list(set([atom_idx] + [mol.GetBondWithIdx(b).GetBeginAtomIdx() for b in bonds] + 
                                   [mol.GetBondWithIdx(b).GetEndAtomIdx() for b in bonds]))
                    
                    fragment_smiles = self._extract_smiles_from_highlighted_atoms(mol, atoms)
                    
                    if fragment_smiles:
                        fragments_info['common_smiles'].append(fragment_smiles)
                        fragments_info['molecule_highlights'].append({
                            'mol': mol, 
                            'highlight_atoms': atoms, 
                            'highlight_bonds': bonds, 
                            'parent_info': mol_info, 
                            'fragment_smiles': fragment_smiles
                        })
                        fragment_found_for_mol = True
                        successful_extractions += 1
                        break
                
                if not fragment_found_for_mol:
                    highlight_only_count += 1
                    try:
                        atom_idx, radius = bit_info[feat_idx][0]
                        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx) or []
                        bonds = list(env)
                        atoms = list(set([atom_idx] + [mol.GetBondWithIdx(b).GetBeginAtomIdx() for b in bonds] + 
                                       [mol.GetBondWithIdx(b).GetEndAtomIdx() for b in bonds]))
                        fragments_info['molecule_highlights'].append({
                            'mol': mol, 
                            'highlight_atoms': atoms, 
                            'highlight_bonds': bonds, 
                            'parent_info': mol_info, 
                            'fragment_smiles': "Highlight_Only"
                        })
                    except:
                        pass
        
        progress_container.empty()
        status_container.empty()
        
        if fragments_info['common_smiles']: 
            fragments_info['most_common'] = Counter(fragments_info['common_smiles']).most_common(5)
            st.success(f"✅ **Feature {feat_idx}**: Extracted {successful_extractions} fragment SMILES, {highlight_only_count} highlights only")
            
            with st.expander("🧪 Fragment Extraction Results", expanded=False):
                for i, (frag_smiles, count) in enumerate(fragments_info['most_common'][:3]):
                    st.write(f"**{i+1}.** `{frag_smiles}` (found {count} times)")
        elif fragments_info['molecule_highlights']: 
            fragments_info['most_common'] = [("No SMILES Extracted", len(fragments_info['molecule_highlights']))]
            st.warning(f"⚠️ **Feature {feat_idx}**: No fragment SMILES extracted, but created {len(fragments_info['molecule_highlights'])} highlights")
        
        return fragments_info

    def _extract_smiles_from_highlighted_atoms(self, mol, highlighted_atoms):
        """Extract SMILES from highlighted atoms."""
        if not highlighted_atoms:
            return None
        
        # Method 1: Connected subgraph approach
        try:
            bonds_to_include = []
            for bond in mol.GetBonds():
                if (bond.GetBeginAtomIdx() in highlighted_atoms and 
                    bond.GetEndAtomIdx() in highlighted_atoms):
                    bonds_to_include.append(bond.GetIdx())
            
            if bonds_to_include:
                submol = Chem.PathToSubmol(mol, bonds_to_include)
                if submol and submol.GetNumAtoms() > 0:
                    try:
                        Chem.SanitizeMol(submol)
                        fragment_smiles = Chem.MolToSmiles(submol, canonical=True)
                        if fragment_smiles and len(fragment_smiles) > 1:
                            return fragment_smiles
                    except:
                        pass
        except:
            pass
        
        # Method 2: Empirical formula approach
        try:
            element_counts = {}
            for atom_idx in highlighted_atoms:
                atom = mol.GetAtomWithIdx(atom_idx)
                symbol = atom.GetSymbol()
                element_counts[symbol] = element_counts.get(symbol, 0) + 1
            
            formula_parts = []
            for element in ['C', 'N', 'O', 'F', 'Cl', 'Br', 'I', 'S', 'P']:
                if element in element_counts:
                    count = element_counts[element]
                    if count == 1:
                        formula_parts.append(element)
                    else:
                        formula_parts.append(f"{element}{count}")
            
            if formula_parts:
                return "".join(formula_parts)
        except:
            pass
        
        # Method 3: Last resort
        try:
            return f"Fragment_{len(highlighted_atoms)}_atoms"
        except:
            pass
        
        return None

    def _create_fallback_gallery(self, frags_info):
        """Create fallback display when fragment extraction fails."""
        num_highlights = len(frags_info.get('molecule_highlights', []))
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, f"Important Feature Detected\n\nFound in {num_highlights} molecules\n\n(Fragment structure could not\nbe automatically extracted)", 
               ha='center', va='center', fontsize=10, 
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="orange"))
        ax.axis('off')
        return fig

    def get_comprehensive_fragment_gallery(self, frags_info):
        """Create gallery of fragment structures."""
        if not frags_info.get('most_common') or frags_info['most_common'][0][0] in ["No SMILES Extracted", "Highlight_Only"]:
            return self._create_fallback_gallery(frags_info)
        
        mols, legends = [], []
        for frag_smiles, count in frags_info['most_common']:
            try:
                mol = Chem.MolFromSmiles(frag_smiles)
                if mol: 
                    mols.append(mol)
                    legends.append(f"Count: {count}\n{frag_smiles}")
            except:
                pass
        
        if mols:
            return Draw.MolsToGridImage(mols, molsPerRow=min(3, len(mols)), subImgSize=(200, 200), 
                                      legends=legends, useSVG=False)
        else:
            return self._create_fallback_gallery(frags_info)

    def get_integrated_fragment_display(self, frags_info, max_examples=3):
        """Get integrated fragment display."""
        if not frags_info.get('most_common'):
            return None, None
        
        most_common_frag_smiles = frags_info['most_common'][0][0]
        
        if most_common_frag_smiles in ["No SMILES Extracted", "No_Fragment_SMILES", "Highlight_Only"]:
            return None, None
        
        matching_highlights = [
            h for h in frags_info['molecule_highlights'] 
            if h.get('fragment_smiles') == most_common_frag_smiles
        ][:max_examples]
        
        isolated_img = None
        try:
            frag_mol = Chem.MolFromSmiles(most_common_frag_smiles)
            if frag_mol:
                isolated_img = Draw.MolToImage(frag_mol, size=(200, 200))
            else:
                isolated_img = self._create_text_fragment_image(most_common_frag_smiles)
        except:
            isolated_img = self._create_text_fragment_image(most_common_frag_smiles)
        
        parent_images = []
        for highlight_info in matching_highlights:
            parent_img = create_molecule_aura_image(
                highlight_info['mol'], 
                highlight_info['highlight_atoms'], 
                highlight_info['highlight_bonds']
            )
            if parent_img:
                parent_images.append({
                    'image': parent_img,
                    'affinity': highlight_info['parent_info']['affinity'],
                    'smiles': highlight_info['parent_info']['smiles']
                })
        
        return isolated_img, parent_images

    def _create_text_fragment_image(self, fragment_text):
        """Create text-based image for fragments."""
        try:
            fig, ax = plt.subplots(figsize=(3, 2))
            
            if fragment_text.startswith("Fragment_") and "atoms" in fragment_text:
                display_text = f"Chemical Fragment\n\n{fragment_text.replace('_', ' ')}"
                color = "lightblue"
            elif any(char.isdigit() for char in fragment_text) and len(fragment_text) < 20:
                display_text = f"Molecular Formula\n\n{fragment_text}"
                color = "lightgreen"
            else:
                display_text = f"Fragment Pattern\n\n{fragment_text}"
                color = "lightyellow"
            
            ax.text(0.5, 0.5, display_text, 
                ha='center', va='center', fontsize=10, weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, edgecolor="gray"))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title("Fragment Representation", fontsize=12, weight='bold')
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
            buf.seek(0)
            img = Image.open(buf)
            plt.close(fig)
            
            return img
        except:
            return None

    def get_affinity_plot_fig(self, feat_data):
        """Create affinity distribution plot."""
        affinities = [m['affinity'] for m in feat_data['molecules_with_feature']]
        if len(affinities) < 3: return None
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(affinities, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        mean, std = np.mean(affinities), np.std(affinities)
        ax.axvline(mean, color='red', ls='--', label=f'Mean: {mean:.2f} ± {std:.2f}')
        ax.set_title(f'Affinity (Feat. {feat_data["feature_index"]})')
        ax.set_xlabel('Binding Affinity')
        ax.set_ylabel('Frequency')
        ax.legend()
        plt.tight_layout()
        return fig

class MolecularDesignTemplateGenerator:
    def generate_scaffolds(self, target):
        if target == "TYK2": 
            return {
                'quinazoline': {'smiles': 'c1cc(F)cc2c1ncnc2Cl', 'name': 'Fluorochloroquinazoline'}, 
                'pyrimidine': {'smiles': 'c1nc(N)nc(c(F)c(F)c1)c2ccccc2F', 'name': 'Trifluoropyrimidine'}
            }
        elif target == "USP7": 
            return {
                'quinoxaline': {'smiles': 'O=C(N1CCN(CC1)c2cnc3ccccc3n2)c4cccnc4', 'name': 'Carbonyl quinoxaline'}, 
                'pyrazine': {'smiles': 'CCN(CC)c1cnc(nc1)C(=O)Nc2ccc3ccccc3c2', 'name': 'Pyrazine diethylamine'}
            }
        return {}
    
    def calculate_properties(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return {}
        return {
            'MW': Descriptors.MolWt(mol), 
            'LogP': Descriptors.MolLogP(mol), 
            'HBD': Descriptors.NumHDonors(mol), 
            'HBA': Descriptors.NumHAcceptors(mol), 
            'TPSA': Descriptors.TPSA(mol), 
            'RotBonds': Descriptors.NumRotatableBonds(mol)
        }
    
    def get_scaffold_grid_image(self, scaffolds):
        mols = [Chem.MolFromSmiles(data['smiles']) for data in scaffolds.values()]
        legends = [data['name'] for data in scaffolds.values()]
        return Draw.MolsToGridImage(mols, molsPerRow=2, subImgSize=(300, 300), 
                                  legends=legends, useSVG=False) if mols else None

class PublicationFigureGenerator:
    def __init__(self, fragment_mappings, datasets):
        self.fragment_mappings = fragment_mappings
        self.datasets = datasets
    
    def _get_representative_molecules(self, target, protocol, n_examples=3):
        mapper, fragments = self.fragment_mappings.get(f"{target}_{protocol}", (None, None))
        if not fragments: return []
        
        examples = []
        used_molecules = set()
        sorted_features = sorted(fragments.items(), key=lambda item: item[1]['average_importance'], reverse=True)
        
        for feature_idx, feature_data in sorted_features:
            if len(examples) >= n_examples: break
            if 'molecule_highlights' in feature_data['fragments']:
                for highlight_info in feature_data['fragments']['molecule_highlights']:
                    if highlight_info['parent_info']['smiles'] not in used_molecules:
                        examples.append({
                            **highlight_info, 
                            'importance': feature_data['average_importance'], 
                            'feature_idx': feature_idx
                        })
                        used_molecules.add(highlight_info['parent_info']['smiles'])
                        break
        return examples
    
    def create_main_figure(self, target, protocol):
        fig = plt.figure(figsize=(15, 5))
        grid_spec = GridSpec(1, 3, figure=fig, wspace=0.1)
        examples = self._get_representative_molecules(target, protocol, 3)
        
        if not examples: return None
        
        for i, example in enumerate(examples):
            ax = fig.add_subplot(grid_spec[0, i])
            aura_image = create_molecule_aura_image(example['mol'], example['highlight_atoms'], example['highlight_bonds'])
            if aura_image: ax.imshow(aura_image)
            ax.axis('off')
            ax.set_title(f"Feat. {example['feature_idx']} | SHAP: {example['importance']:.3f}\nAffinity: {example['parent_info']['affinity']:.2f}", fontsize=10)
        
        fig.suptitle(f"Key SHAP-Identified Features for {target} ({protocol})", fontsize=16, weight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

class SARAnalyzer:
    def __init__(self, dataset_df):
        self.df = dataset_df.copy()
        self.df['mol'] = self.df['SMILES'].apply(Chem.MolFromSmiles)
        self.df.dropna(subset=['mol'], inplace=True)
        self.df['mol_id'] = self.df.index.astype(str)
    
    def analyze_r_groups(self):
        scaffolds = [MurckoScaffold.GetScaffoldForMol(m) for m in self.df['mol']]
        most_common_scaffold_smarts = Counter([Chem.MolToSmarts(s) for s in scaffolds]).most_common(1)[0][0]
        core = Chem.MolFromSmarts(most_common_scaffold_smarts)
        decomp, unmatched = rdRGroupDecomposition.RGroupDecompose([core], list(self.df['mol']), asSmiles=True)
        r_group_df = pd.DataFrame(decomp).join(self.df.reset_index())
        return core, r_group_df
    
    def analyze_activity_cliffs(self, activity_threshold=1.0):
        """Analyze activity cliffs using molecular similarity."""
        cliffs = []
        
        fps = []
        for idx, row in self.df.iterrows():
            mol = row['mol']
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
                fps.append((row['mol_id'], fp, row['affinity'], mol))
            except:
                continue
        
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                id1, fp1, act1, mol1 = fps[i]
                id2, fp2, act2, mol2 = fps[j]
                
                similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
                activity_delta = abs(act1 - act2)
                
                if similarity > 0.7 and activity_delta >= activity_threshold:
                    cliffs.append({
                        'mol1': mol1,
                        'mol2': mol2,
                        'act1': act1,
                        'act2': act2,
                        'delta': activity_delta,
                        'similarity': similarity,
                        'transform': f"Similar structures (Sim: {similarity:.2f})"
                    })
        
        return sorted(cliffs, key=lambda x: x['delta'], reverse=True)[:20]

# ==============================================================================
# 5. VISUALIZATION FUNCTIONS
# ==============================================================================

def create_similarity_distribution_plot(all_compounds_df, datasets_to_plot, filename='similarity_dist.png', sample_size=1500):
    """
    Creates a 1x4 similarity distribution plot for all datasets.
    Each subplot shows the similarity distribution using different fingerprint methods.
    """
    print(f"Generating similarity plot for: {datasets_to_plot}")
    
    # Set up 1x4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharex=True)
    
    # If only one subplot, make it a list for consistency
    if len(datasets_to_plot) == 1:
        axes = [axes]

    # Define fingerprint configurations
    repr_configs = [
        ('ECFP', smiles_to_ecfp, 'jaccard'),
        ('MACCS', smiles_to_maccs, 'jaccard'),
        ('ChemBERTa', smiles_to_chemberta, 'cosine')
    ]

    # Track maximum density across all subplots for consistent y-axis scaling
    max_density = 0
    all_densities = {}
    x_range = np.linspace(0, 1, 200)
    
    for i, dataset in enumerate(datasets_to_plot):
        dataset_df = all_compounds_df[all_compounds_df['Dataset'] == dataset]
        sampled_df = dataset_df.sample(min(sample_size, len(dataset_df)), random_state=42)
        
        all_densities[dataset] = {}
        
        for repr_name, calc_func, metric in repr_configs:
            # Calculate fingerprints and similarities
            fingerprints = calc_func(sampled_df)
            similarities = calculate_similarity_matrix_safe(fingerprints, metric)
            
            if len(similarities) > 10:
                # Calculate density
                kde = gaussian_kde(similarities)
                kde.set_bandwidth(kde.factor * 0.7)
                density = kde(x_range)
                all_densities[dataset][repr_name] = density
                max_density = max(max_density, np.max(density))

    # Second pass: create the plots with consistent y-axis
    for i, dataset in enumerate(datasets_to_plot):
        ax = axes[i]
        
        for repr_name, calc_func, metric in repr_configs:
            if repr_name in all_densities[dataset]:
                density = all_densities[dataset][repr_name]
                line = ax.plot(x_range, density, color=FP_COLORS[repr_name], 
                             linewidth=3, alpha=0.9, label=repr_name)[0]
                ax.fill_between(x_range, density, color=FP_COLORS[repr_name], alpha=0.1)

        # Customize subplot
        ax.set_title(dataset, fontsize=FONT_SIZES['title'], fontweight='bold')
        ax.set_xlabel('Similarity', fontsize=FONT_SIZES['label'])
        
        # Add y-label only to leftmost plot
        if i == 0:
            ax.set_ylabel('Density', fontsize=FONT_SIZES['label'])
        
        # Set consistent axis limits with some padding
        ax.set_xlim(0, 1)
        ax.set_ylim(0, max_density * 1.05)  # Add 5% padding at the top
        
        # Add legend to first subplot only
        if i == 0:
            ax.legend(fontsize=FONT_SIZES['legend'], frameon=True)
        
        apply_plot_style(ax)

    # Final layout adjustments
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle('', fontsize=FONT_SIZES['title']+4, fontweight='bold')
    
    plt.savefig(filename, dpi=500, bbox_inches='tight')
    print(f"Saved figure: {filename}")
    plt.close(fig)
    
    return filename

def calculate_molecular_properties(df):
    """Calculate key molecular properties for all compounds."""
    print("Calculating Molecular Properties...")
    from tqdm import tqdm
    
    props_list = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Calculating Properties"):
        mol = Chem.MolFromSmiles(row['SMILES'])
        props = {'Dataset': row['Dataset']}
        if 'affinity' in row: 
            props['Affinity'] = row['affinity']
        
        if mol:
            props['MW'] = Descriptors.MolWt(mol)
            props['LogP'] = Descriptors.MolLogP(mol)
            props['TPSA'] = Descriptors.TPSA(mol)
            props['HBD'] = Descriptors.NumHDonors(mol)
            props['HBA'] = Descriptors.NumHAcceptors(mol)
            props['RotBonds'] = Descriptors.NumRotatableBonds(mol)
        props_list.append(props)
        
    return pd.DataFrame(props_list).dropna()

def create_property_distribution_boxplots(props_df, datasets_to_plot, filename='property_boxplots.png'):
    """Creates boxplots for key molecular properties for all datasets."""
    print(f"Generating property boxplots for: {datasets_to_plot}")
    
    properties_to_plot = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds']
    property_labels = {
        'MW': 'Mol. Weight (Da)', 'LogP': 'LogP', 'TPSA': 'TPSA (Å²)',
        'HBD': 'H-Bond Donors', 'HBA': 'H-Bond Acceptors', 'RotBonds': 'Rotatable Bonds'
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    df_subset = props_df[props_df['Dataset'].isin(datasets_to_plot)]

    for i, prop in enumerate(properties_to_plot):
        ax = axes[i]
        sns.boxplot(
            x='Dataset', y=prop, data=df_subset, ax=ax,
            order=datasets_to_plot, palette=[DATASET_COLORS[d] for d in datasets_to_plot],
            fliersize=2, width=0.6
        )
        
        ax.set_title(property_labels[prop], fontsize=FONT_SIZES['title'], fontweight='bold')
        ax.set_ylabel('')
        ax.set_xlabel('')
        apply_plot_style(ax)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle('', fontsize=FONT_SIZES['title']+4, fontweight='bold')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {filename}")
    plt.close(fig)
    
    return filename

def create_performance_summary_figure(results_df, datasets_to_plot, filename='performance_summary.png'):
    """
    Creates a 3-panel summary of AL performance for all datasets.
    Panels: A) Final Recall Violin Plot, B) Success Rate Bar Plot, C) Learning Curves.
    """
    print(f"Generating performance summary for: {datasets_to_plot}")

    final_perf = results_df.groupby(['Dataset', 'Protocol', 'Kernel', 'Fingerprint', 'Seed']).last().reset_index()
    final_perf = final_perf[final_perf['Dataset'].isin(datasets_to_plot)]
    results_df = results_df[results_df['Dataset'].isin(datasets_to_plot)]

    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])

    # Panel A: Performance Distribution (Violin Plot)
    ax1 = fig.add_subplot(gs[0, 0])
    violin_data = [final_perf[final_perf['Dataset'] == d]['Recall (2%)'] for d in datasets_to_plot]
    parts = ax1.violinplot(violin_data, showmeans=True, showmedians=False, widths=0.8)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(DATASET_COLORS[datasets_to_plot[i]])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
    parts['cmeans'].set_color('black')
    
    ax1.set_xticks(np.arange(1, len(datasets_to_plot) + 1))
    ax1.set_xticklabels(datasets_to_plot)
    ax1.set_ylabel('Final Recall (Top 2%)', fontsize=FONT_SIZES['label'])
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title('A) Final Performance Distribution', fontsize=FONT_SIZES['title'], fontweight='bold', loc='left')
    apply_plot_style(ax1)

    # Panel B: Success Rate Analysis
    ax2 = fig.add_subplot(gs[0, 1])
    threshold = 0.5
    success_rates = []
    for d in datasets_to_plot:
        data = final_perf[final_perf['Dataset'] == d]
        rate = (data['Recall (2%)'] > threshold).sum() / len(data) * 100 if len(data) > 0 else 0
        success_rates.append(rate)
        
    ax2.bar(datasets_to_plot, success_rates, color=[DATASET_COLORS[d] for d in datasets_to_plot], alpha=0.8, edgecolor='black')
    for i, rate in enumerate(success_rates):
        ax2.text(i, rate + 2, f'{rate:.1f}%', ha='center', fontsize=FONT_SIZES['annotation'], fontweight='bold')
    
    ax2.set_ylabel(f'% of Methods with Recall > {threshold*100:.0f}%', fontsize=FONT_SIZES['label'])
    ax2.set_ylim(0, 105)
    ax2.set_title('B) Success Rate Analysis', fontsize=FONT_SIZES['title'], fontweight='bold', loc='left')
    apply_plot_style(ax2)

    # Panel C: Learning Curves by Dataset
    ax3 = fig.add_subplot(gs[1, :])
    for dataset in datasets_to_plot:
        data = results_df[results_df['Dataset'] == dataset]
        non_random = data[data['Protocol'] != 'random']
        if not non_random.empty:
            summary = non_random.groupby('Cycle')['Recall (2%)'].agg(['mean', 'std']).reset_index()
            ax3.plot(summary['Cycle'], summary['mean'], 'o-', 
                     color=DATASET_COLORS[dataset], linewidth=3, markersize=8, label=dataset)
            ax3.fill_between(summary['Cycle'], summary['mean'] - summary['std'], summary['mean'] + summary['std'], 
                             alpha=0.15, color=DATASET_COLORS[dataset])
    
    ax3.set_xlabel('Active Learning Cycle', fontsize=FONT_SIZES['label'])
    ax3.set_ylabel('Mean Recall (Top 2%)', fontsize=FONT_SIZES['label'])
    ax3.set_ylim(0, 0.8)
    ax3.legend(fontsize=FONT_SIZES['legend'], title='Dataset', title_fontsize=FONT_SIZES['legend'])
    ax3.set_title('C) Learning Progress', fontsize=FONT_SIZES['title'], fontweight='bold', loc='left')
    apply_plot_style(ax3)
    
    plt.tight_layout(pad=2.0)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {filename}")
    plt.close(fig)
    
    return filename

def create_full_performance_heatmap(df, filename='performance_heatmap.png', metric='Recall (2%)'):
    """Creates a 2x2 heatmap of final performance for ALL datasets."""
    print(f"Generating full performance heatmap for all datasets.")
    
    final_data = df[df['Cycle'] == 10].groupby(['Dataset', 'Protocol', 'Kernel', 'Fingerprint'])[metric].mean().reset_index()
    datasets = ['TYK2', 'USP7', 'D2R', 'MPRO']  # Ensure consistent order

    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    axes = axes.flatten()
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        dataset_data = final_data[final_data['Dataset'] == dataset]
        
        if dataset_data.empty:
            continue
            
        pivot_data = dataset_data.pivot_table(
            values=metric,
            index='Protocol', 
            columns=['Kernel', 'Fingerprint'],
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='viridis', 
                   ax=ax, cbar_kws={'label': f'Mean {metric}'},
                   linewidths=0.5, annot_kws={"size": FONT_SIZES['annotation']-2})
        
        ax.set_title(dataset, fontsize=FONT_SIZES['title'], fontweight='bold')
        ax.set_xlabel('Kernel & Fingerprint', fontsize=FONT_SIZES['label'])
        ax.set_ylabel('Protocol', fontsize=FONT_SIZES['label'])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        apply_plot_style(ax)

    fig.suptitle(f'', fontsize=FONT_SIZES['title']+4, y=1.0, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {filename}")
    plt.close(fig)
    
    return filename

def create_color_picker(label, num_colors):
    """Create color pickers for plot elements."""
    st.subheader(f"Colors for {label}")
    colors = []
    cols = st.columns(4)
    
    default_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
        '#bcbd22', '#17becf', '#ff9896', '#98df8a', 
        '#c5b0d5', '#c49c94', '#f7b6d2'
    ]
    
    for i in range(num_colors):
        with cols[i % 4]:
            default_color = default_colors[i % len(default_colors)]
            color = st.color_picker(
                f'{label} Color {i+1}',
                value=default_color
            )
            colors.append(color)
    return colors

def create_protocol_comparison_plot(df, protocol, datasets, kernels, fingerprints, colors, plot_config):
    """Create performance plot for a specific protocol."""
    set_plot_style(plot_config)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Performance Comparison - {protocol}', y=1.02)
    
    linestyles = ['-', '--', ':', '-.']
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx//2, idx%2]
        dataset_data = df[
            (df['Dataset'] == dataset) &
            (df['Protocol'] == protocol)
        ]
        
        for k_idx, kernel in enumerate(kernels):
            for f_idx, fingerprint in enumerate(fingerprints):
                data = dataset_data[
                    (dataset_data['Kernel'].str.contains(kernel)) &
                    (dataset_data['Fingerprint'] == fingerprint)
                ]
                
                if not data.empty:
                    grouped = data.groupby('Compounds acquired')['Recall (2%)']
                    mean = grouped.mean()
                    std = grouped.std()
                    
                    ax.plot(mean.index, mean.values,
                           label=f'{kernel}-{fingerprint}',
                           color=colors[k_idx],
                           linestyle=linestyles[f_idx % len(linestyles)],
                           linewidth=2)
                    
                    if plot_config['show_error_bars']:
                        ax.fill_between(mean.index,
                                      mean.values - std.values,
                                      mean.values + std.values,
                                      alpha=0.15,
                                      color=colors[k_idx])
        
        ax.set_title(dataset, pad=20)
        ax.set_xlabel('Compounds acquired')
        ax.set_ylabel('Recall (2%)')
        ax.set_ylim(bottom=0)
        
        if plot_config['show_grid']:
            ax.grid(True, linestyle='--', alpha=0.3)
        
        if idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    
    plt.tight_layout()
    return fig

def create_dataset_performance_plot(df, mw_df, dataset, kernels, fingerprints, colors, plot_config):
    """Create performance plot showing MW and Random baselines."""
    set_plot_style(plot_config)
    
    protocols = sorted([p for p in df['Protocol'].unique() if p not in ['random']])
    
    fingerprint_styles = {
        'chemberta': '-',
        'ecfp': '--',
        'maccs': ':'
    }
    
    mw_baseline = mw_df[mw_df['Dataset'] == dataset] if not mw_df.empty else pd.DataFrame()
    random_baseline = df[
        (df['Dataset'] == dataset) &
        (df['Protocol'] == 'random')
    ]
    
    n_protocols = len(protocols)
    n_cols = min(2, n_protocols)
    n_rows = (n_protocols + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(8*n_cols, 6*n_rows)
    )
    
    if n_protocols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for idx, protocol in enumerate(protocols):
        ax = axes[idx]
        
        # Plot Random baseline
        if not random_baseline.empty:
            x_values = sorted(random_baseline['Compounds acquired'].unique())
            random_means = []
            random_stds = []
            
            for x in x_values:
                data_at_x = random_baseline[random_baseline['Compounds acquired'] == x]['Recall (2%)']
                random_means.append(data_at_x.mean())
                random_stds.append(data_at_x.std())
            
            ax.fill_between(
                x_values,
                [m - s for m, s in zip(random_means, random_stds)],
                [m + s for m, s in zip(random_means, random_stds)],
                color='gray',
                alpha=0.15,
                label='Random (±σ)'
            )
            ax.plot(
                x_values,
                random_means,
                color='gray',
                linestyle=':',
                linewidth=1.5,
                label='Random (mean)',
                alpha=0.8
            )
        
        # Plot MW baseline
        if not mw_baseline.empty:
            x_values_mw = sorted(mw_baseline['Compounds_acquired'].unique())
            mw_means = []
            mw_stds = []
            
            for x in x_values_mw:
                data_at_x = mw_baseline[mw_baseline['Compounds_acquired'] == x]['Recall (2%)']
                mw_means.append(data_at_x.mean())
                mw_stds.append(data_at_x.std())
            
            ax.fill_between(
                x_values_mw,
                [m - s for m, s in zip(mw_means, mw_stds)],
                [m + s for m, s in zip(mw_means, mw_stds)],
                color='red',
                alpha=0.15,
                label='MW (±σ)'
            )
            ax.plot(
                x_values_mw,
                mw_means,
                color='red',
                linestyle=':',
                linewidth=1.5,
                label='MW (mean)',
                alpha=0.8
            )
        
        # Plot protocol data
        protocol_data = df[
            (df['Dataset'] == dataset) &
            (df['Protocol'] == protocol)
        ]
        
        color_idx = 0
        for kernel in kernels:
            for fingerprint in fingerprints:
                data = protocol_data[
                    (protocol_data['Kernel'].str.contains(kernel)) &
                    (protocol_data['Fingerprint'] == fingerprint)
                ]
                
                if not data.empty:
                    grouped = data.groupby('Compounds acquired')['Recall (2%)']
                    mean = grouped.mean()
                    std = grouped.std()
                    
                    ax.plot(
                        mean.index,
                        mean.values,
                        label=f'{kernel}-{fingerprint}',
                        color=colors[color_idx],
                        linestyle=fingerprint_styles.get(fingerprint, '-'),
                        linewidth=2,
                        alpha=0.8
                    )
                    
                    if plot_config['show_error_bars']:
                        ax.fill_between(
                            mean.index,
                            mean.values - std.values,
                            mean.values + std.values,
                            alpha=0.1,
                            color=colors[color_idx]
                        )
                    color_idx += 1
        
        for y in [0.2, 0.4, 0.6]:
            ax.axhline(y=y, color='gray', linestyle=':', alpha=0.3)
        
        ax.set_title(f'{protocol}', pad=10, fontsize=plot_config['label_size'])
        ax.set_xlabel('Compounds acquired', fontsize=plot_config['label_size'])
        ax.set_ylabel('Recall (2%)', fontsize=plot_config['label_size'])
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=plot_config['tick_size'])
        
        if plot_config['show_grid']:
            ax.grid(True, linestyle='--', alpha=0.2)
        
        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()
            baseline_indices = [i for i, label in enumerate(labels) if 'Random' in label or 'MW' in label]
            for i in reversed(baseline_indices):
                handles.insert(0, handles.pop(i))
                labels.insert(0, labels.pop(i))
            
            ax.legend(
                handles,
                labels,
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                fontsize='small',
                title='Kernel-Fingerprint'
            )
    
    for idx in range(len(protocols), len(axes)):
        fig.delaxes(axes[idx])
    
    fig.suptitle(f'{dataset} Performance Across Protocols', 
                y=1.02, 
                fontsize=plot_config['title_size'])
    plt.tight_layout()
    
    return fig

def create_kernel_performance_plot(df, kernel, datasets, protocols, fingerprints, colors, plot_config):
    """Create performance plot for a specific kernel."""
    set_plot_style(plot_config)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Protocol Performance - {kernel} Kernel', y=1.02)

    bar_width = 0.8 / len(fingerprints)

    for idx, dataset in enumerate(datasets):
        ax = axes[idx//2, idx%2]
        x = np.arange(len(protocols))

        for f_idx, fingerprint in enumerate(fingerprints):
            recalls = []
            errors = []

            for protocol in protocols:
                data = df[
                    (df['Dataset'] == dataset) &
                    (df['Kernel'].str.contains(kernel)) &
                    (df['Fingerprint'] == fingerprint) &
                    (df['Protocol'] == protocol)
                ]

                if not data.empty:
                    max_compounds = data['Compounds acquired'].max()
                    final_data = data[data['Compounds acquired'] == max_compounds]
                    recalls.append(final_data['Recall (2%)'].mean())
                    errors.append(final_data['Recall (2%)'].std())
                else:
                    recalls.append(0)
                    errors.append(0)

            ax.bar(x + f_idx * bar_width, recalls, bar_width,
                  label=fingerprint,
                  color=colors[f_idx],
                  alpha=0.8)

            if plot_config['show_error_bars']:
                ax.errorbar(x + f_idx * bar_width, recalls, yerr=errors,
                          fmt='none', color='black', capsize=3)

        ax.set_title(dataset, pad=20)
        ax.set_xticks(x + bar_width * (len(fingerprints) - 1) / 2)
        ax.set_xticklabels(protocols, rotation=45, ha='right')
        ax.set_ylabel('Final Recall (2%)')

        if plot_config['show_grid']:
            ax.grid(True, linestyle='--', alpha=0.3, axis='y')

        if idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

    plt.tight_layout()
    return fig

def create_fingerprint_performance_plot(df, fingerprint, datasets, protocols, kernels, colors, plot_config):
    """Create performance plot for a specific fingerprint."""
    set_plot_style(plot_config)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Kernel Performance - {fingerprint} Fingerprint', y=1.02)

    bar_width = 0.8 / len(kernels)

    for idx, dataset in enumerate(datasets):
        ax = axes[idx//2, idx%2]
        x = np.arange(len(protocols))

        for k_idx, kernel in enumerate(kernels):
            recalls = []
            errors = []

            for protocol in protocols:
                data = df[
                    (df['Dataset'] == dataset) &
                    (df['Kernel'].str.contains(kernel)) &
                    (df['Fingerprint'] == fingerprint) &
                    (df['Protocol'] == protocol)
                ]

                if not data.empty:
                    max_compounds = data['Compounds acquired'].max()
                    final_data = data[data['Compounds acquired'] == max_compounds]
                    recalls.append(final_data['Recall (2%)'].mean())
                    errors.append(final_data['Recall (2%)'].std())
                else:
                    recalls.append(0)
                    errors.append(0)

            ax.bar(x + k_idx * bar_width, recalls, bar_width,
                  label=kernel,
                  color=colors[k_idx],
                  alpha=0.8)

            if plot_config['show_error_bars']:
                ax.errorbar(x + k_idx * bar_width, recalls, yerr=errors,
                          fmt='none', color='black', capsize=3)

        ax.set_title(dataset, pad=20)
        ax.set_xticks(x + bar_width * (len(kernels) - 1) / 2)
        ax.set_xticklabels(protocols, rotation=45, ha='right')
        ax.set_ylabel('Final Recall (2%)')

        if plot_config['show_grid']:
            ax.grid(True, linestyle='--', alpha=0.3, axis='y')

        if idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

    plt.tight_layout()
    return fig


def create_ridge_plot_v3(df, mw_df, dataset, timepoint=360):
    """Create vertical ridge plot."""
    # Get valid timepoint
    available = sorted(df['Compounds acquired'].unique())
    if timepoint not in available:
        timepoint = available[-1]
    
    # Calculate protocol metrics
    df_t = df[df['Compounds acquired'] == timepoint]
    metrics = {}
    for protocol in df_t['Protocol'].unique():
        if protocol != 'random':
            data = df_t[df_t['Protocol'] == protocol]['Recall (2%)']
            metrics[protocol] = data.mean()
    
    protocols = sorted(metrics.keys(), key=lambda x: metrics[x])
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(12, max(8, len(protocols) * 0.8)))
    
    # Colors
    protocol_colors = plt.cm.viridis(np.linspace(0, 0.9, len(protocols)))
    baseline_colors = {'random': '#808080', 'mw': '#FF0000'}
    
    spacing = 1.0
    positions = {p: i*spacing for i, p in enumerate(['random', 'mw'] + protocols)}
    
    # Plot Random baseline
    random_data = df[
        (df['Dataset'] == dataset) &
        (df['Protocol'] == 'random') &
        (df['Compounds acquired'] == timepoint)
    ]['Recall (2%)']
    
    if len(random_data) > 1:
        kde = gaussian_kde(random_data)
        x_range = np.linspace(random_data.min(), random_data.max(), 200)
        density = kde(x_range)
        scaled_density = density / density.max() * spacing * 0.8
        ax.fill_between(x_range,
                        positions['random'],
                        positions['random'] + scaled_density,
                        alpha=0.8,
                        color=baseline_colors['random'],
                        label='Random')
        ax.vlines(random_data.mean(),
                 positions['random'],
                 positions['random'] + scaled_density.max(),
                 color='white',
                 linewidth=2)
    
    # Plot MW baseline
    if not mw_df.empty:
        mw_data = mw_df[
            (mw_df['Dataset'] == dataset) &
            (mw_df['Compounds_acquired'] == timepoint)
        ]['Recall (2%)']
        
        if len(mw_data) > 1:
            kde = gaussian_kde(mw_data)
            x_range = np.linspace(mw_data.min(), mw_data.max(), 200)
            density = kde(x_range)
            scaled_density = density / density.max() * spacing * 0.8
            ax.fill_between(x_range,
                            positions['mw'],
                            positions['mw'] + scaled_density,
                            alpha=0.8,
                            color=baseline_colors['mw'],
                            label='MW')
            ax.vlines(mw_data.mean(),
                     positions['mw'],
                     positions['mw'] + scaled_density.max(),
                     color='white',
                     linewidth=2)
    
    # Plot protocol distributions
    for idx, protocol in enumerate(protocols):
        data = df[
            (df['Dataset'] == dataset) &
            (df['Protocol'] == protocol) &
            (df['Compounds acquired'] == timepoint)
        ]['Recall (2%)']
        
        if len(data) > 1:
            kde = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            density = kde(x_range)
            scaled_density = density / density.max() * spacing * 0.8
            
            ax.fill_between(x_range,
                          positions[protocol],
                          positions[protocol] + scaled_density,
                          alpha=0.8,
                          color=protocol_colors[idx],
                          label=protocol)
            
            ax.vlines(data.mean(),
                     positions[protocol],
                     positions[protocol] + scaled_density.max(),
                     color='white',
                     linewidth=2)
    
    # Add grid lines
    for x in np.arange(0, 1, 0.2):
        ax.axvline(x, color='gray', alpha=0.2, linestyle='--')
    
    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks([positions[p] + 0.5 for p in ['random', 'mw'] + protocols])
    ax.set_yticklabels(['Random', 'MW'] + protocols)
    ax.set_xlabel('Recall (2%)')
    ax.set_title(f'Performance Distributions at {timepoint} Compounds - {dataset}')
    
    plt.tight_layout()
    return fig

def plot_affinity_predictions(smiles_df, cycle=0):
    """Plot actual vs predicted affinity values."""
    pred_col = f'pred_{cycle}'
    
    if pred_col not in smiles_df.columns:
        pred_cols = [col for col in smiles_df.columns if col.startswith('pred_')]
        if not pred_cols:
            st.error("No prediction columns found in SMILES data")
            return None
        pred_col = pred_cols[0]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    plot_df = smiles_df.dropna(subset=['affinity', pred_col])
    
    min_val = min(plot_df['affinity'].min(), plot_df[pred_col].min())
    max_val = max(plot_df['affinity'].max(), plot_df[pred_col].max())
    
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    if len(plot_df) > 10:
        try:
            xy = np.vstack([plot_df['affinity'], plot_df[pred_col]])
            z = gaussian_kde(xy)(xy)
            
            scatter = ax.scatter(
                plot_df['affinity'], 
                plot_df[pred_col],
                c=z, 
                s=50, 
                alpha=0.7,
                cmap='viridis',
                edgecolor='k',
                linewidths=0.5
            )
            
            cbar = plt.colorbar(scatter)
            cbar.set_label('Density', rotation=270, labelpad=20)
            
        except:
            ax.scatter(
                plot_df['affinity'], 
                plot_df[pred_col],
                s=50, 
                alpha=0.7,
                edgecolor='k',
                linewidths=0.5
            )
    else:
        ax.scatter(
            plot_df['affinity'], 
            plot_df[pred_col],
            s=50, 
            alpha=0.7,
            edgecolor='k',
            linewidths=0.5
        )
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        plot_df['affinity'], plot_df[pred_col]
    )
    ax.plot(
        [min_val, max_val], 
        [intercept + slope*min_val, intercept + slope*max_val], 
        'r-', 
        label=f'Regression (R²={r_value**2:.3f})'
    )
    
    ax.set_xlabel('Experimental Affinity', fontsize=14)
    ax.set_ylabel('Predicted Affinity', fontsize=14)
    ax.set_title(f'Prediction Performance at Cycle {cycle}', fontsize=16)
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def visualize_chemical_space(smiles_df, color_by='affinity', method='PCA', fp_type='ECFP', fp_radius=2, fp_bits=2048):
    """Visualize chemical space using a chosen dimensionality reduction method."""
    from rdkit.Chem import rdFingerprintGenerator # <-- ADD THIS LINE
    if 'SMILES' not in smiles_df.columns:
        st.error("SMILES column not found in data")
        return None, None

    st.info(f"Generating chemical space visualization using {method} on {fp_type} fingerprints...")
    
    # Keep track of valid molecules
    valid_indices = [i for i, s in enumerate(smiles_df['SMILES']) if Chem.MolFromSmiles(s) is not None]
    if not valid_indices:
        st.error("No valid molecules found in the provided data.")
        return None, None
        
    valid_smiles_df = smiles_df.iloc[valid_indices].reset_index(drop=True)

    # --- Fingerprint Generation ---
    with st.spinner("Generating fingerprints..."):
        if fp_type == 'ECFP':
            mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=fp_radius, fpSize=fp_bits)
            fps = [mfpgen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in valid_smiles_df['SMILES']]
        elif fp_type == 'MACCS':
            fps = [np.array(MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(s))) for s in valid_smiles_df['SMILES']]
        else:
            mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=fp_radius, fpSize=fp_bits)
            fps = [mfpgen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in valid_smiles_df['SMILES']]
        X = np.array(fps)

    # --- Dimensionality Reduction ---
    with st.spinner(f"Running {method}..."):
        if method == 'PCA':
            reducer = PCA(n_components=2, random_state=42)
            title_prefix = "Principal Component"
        elif method == 't-SNE':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
            title_prefix = "t-SNE Dimension"
        elif method == 'UMAP' and HAS_UMAP:
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(X)-1), min_dist=0.1)
            title_prefix = "UMAP Dimension"
        else: # Fallback
            reducer = PCA(n_components=2, random_state=42)
            title_prefix = "Principal Component"
        
        X_reduced = reducer.fit_transform(X)

    # --- Create DataFrame for Plotting ---
    plot_df = pd.DataFrame({
        'Dim1': X_reduced[:, 0],
        'Dim2': X_reduced[:, 1],
    })
    
    # Merge back other columns from the valid dataframe
    plot_df = pd.concat([plot_df, valid_smiles_df.reset_index(drop=True)], axis=1)

    # --- Create Interactive Plotly Figure ---
    is_numeric_color = pd.api.types.is_numeric_dtype(plot_df[color_by])

    fig = px.scatter(
        plot_df,
        x='Dim1',
        y='Dim2',
        color=color_by,
        color_discrete_map=DATASET_COLORS if color_by == 'Dataset' else None,
        color_continuous_scale=px.colors.sequential.Viridis if is_numeric_color else None,
        hover_name='SMILES',
        hover_data={'Dim1': False, 'Dim2': False, 'SMILES': False, 'Dataset': True, 'affinity': ':.2f'},
        title=f'Chemical Space Visualization ({method} on {fp_type})',
        labels={'Dim1': f'{title_prefix} 1', 'Dim2': f'{title_prefix} 2'}
    )
    
    fig.update_traces(marker=dict(size=5, opacity=0.7))
    fig.update_layout(legend_title_text=color_by)

    return fig, plot_df

def visualize_top_compounds(smiles_df, n=5, sort_by='affinity', ascending=False):
    """Visualize the top N compounds."""
    if 'SMILES' not in smiles_df.columns:
        st.error("SMILES column not found in data")
        return None, None
    
    if sort_by in smiles_df.columns:
        top_df = smiles_df.sort_values(by=sort_by, ascending=ascending).head(n)
    else:
        st.warning(f"Column {sort_by} not found, using affinity instead")
        if 'affinity' in smiles_df.columns:
            top_df = smiles_df.sort_values(by='affinity', ascending=False).head(n)
        else:
            st.error("Could not find appropriate column to sort by")
            return None, None
    
    mols = []
    valid_indices = []
    labels = []
    
    for i, row in top_df.iterrows():
        try:
            mol = Chem.MolFromSmiles(row['SMILES'])
            if mol:
                mols.append(mol)
                valid_indices.append(i)
                
                if 'affinity' in row and pd.notna(row['affinity']):
                     labels.append(f"{sort_by}: {row[sort_by]:.2f}")
                else:
                     labels.append(f"SMILES: {row['SMILES']}")
        except:
            pass
    
    if not mols:
        st.error("No valid molecules found")
        return None, None
    
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=min(5, len(mols)),
        subImgSize=(300, 300),
        legends=labels
    )
    
    return img, top_df.loc[valid_indices]

@st.cache_data(show_spinner="Downloading demo file: {_file_name}...")
def download_and_cache_file(url, _file_name):
    """Downloads a file from a URL, caches it, and returns its content."""
    import requests
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to download {_file_name}: {e}")
        return None

# ==============================================================================
# 6. MAIN STREAMLIT APPLICATION CLASS
# ==============================================================================
class UnifiedDrugDiscoveryApp:
    """
    Unified Drug Discovery Analysis Platform
    
    A comprehensive Streamlit application that combines:
    - SHAP-guided drug discovery analysis
    - Protocol performance visualization
    - Chemical fragment mapping
    - Molecular design templates
    - Publication-quality figure generation
    
    Usage:
    ------
    1. Upload data files in the sidebar:
       - SHAP analysis results (.pkl files)
       - Molecule datasets (.csv files)
       - Protocol results (.csv files)
    
    2. Navigate through tabs to access different analyses:
       - Overview: Summary of loaded data
       - Feature Evolution: SHAP feature importance over cycles
       - Chemical Fragments: Map features to molecular substructures
       - Drug Design: Generate molecular templates
       - Protocol Performance: Compare different protocols
       - Distribution Analysis: Ridge plots and distributions
       - Molecular Analysis: Affinity predictions and chemical space
       - Publication Figures: Generate manuscript-ready figures
       - Advanced Analytics: Cross-target comparisons
       - Publication Gallery: Access pre-formatted publication figures
    
    3. Configure analysis parameters in the sidebar
    4. Export results and figures for publication
    """
    
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize all session state variables."""
        for key in ['analysis_data', 'uploaded_datasets', 'fragment_mappings', 
                   'main_results_df', 'mw_results_df', 'molecular_data_df', 'molecular_data_source']:
            if key not in st.session_state:
                st.session_state[key] = {} if key in ['analysis_data', 'uploaded_datasets', 'fragment_mappings'] else None
    
    def run(self):
        """Main application entry point."""
        st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
        st.markdown(CUSTOM_APP_CSS, unsafe_allow_html=True)
        
        
        st.markdown(f'<h1 class="main-header">{PAGE_TITLE}</h1>', unsafe_allow_html=True)
        
        # Create sidebar
        self._create_sidebar()

        self._prepare_molecular_data()

        
        # Create main interface
        self._create_main_interface()
    
    def _create_sidebar(self):
        """Create sidebar with data upload and settings."""
        st.sidebar.header("📁 Data Upload")
        

        st.sidebar.markdown("New to the app? Load a complete dataset to get started.")
        if st.sidebar.button("🚀 Load Demo Dataset", use_container_width=True, type="primary"):
            self._load_demo_data()
        st.sidebar.markdown("---")
    
        st.sidebar.markdown("Or, upload your own data below:")
        # SHAP Analysis Files
        with st.sidebar.expander("Upload SHAP Analysis (.pkl)", False):
            uploaded_analysis = st.file_uploader("Select .pkl files", type=['pkl'], 
                                               accept_multiple_files=True, label_visibility="collapsed")
            if uploaded_analysis:
                self._process_files(uploaded_analysis, 'analysis_data', pickle.load)
        
        # Dataset CSV Files  
        with st.sidebar.expander("Upload Molecule Datasets (.csv)", False):
            uploaded_datasets = st.file_uploader("Select .csv files", type=['csv'], 
                                               accept_multiple_files=True, label_visibility="collapsed")
            if uploaded_datasets:
                self._process_files(uploaded_datasets, 'uploaded_datasets', pd.read_csv, by_target=True)
        
        # Protocol Results CSV
        with st.sidebar.expander("Upload Protocol Results (.csv)", False):
            st.write("Main results CSV file")
            main_file = st.file_uploader("Upload main results CSV", type=['csv'], key='main_csv')
            if main_file:
                try:
                    st.session_state.main_results_df = pd.read_csv(main_file)
                    st.sidebar.success("Main results loaded")
                except Exception as e:
                    st.sidebar.error(f"Error: {str(e)}")
            
            st.write("MW baseline CSV file")
            mw_file = st.file_uploader("Upload MW baseline CSV", type=['csv'], key='mw_csv')
            if mw_file:
                try:
                    st.session_state.mw_results_df = pd.read_csv(mw_file)
                    st.sidebar.success("MW baseline loaded")
                except Exception as e:
                    st.sidebar.error(f"Error: {str(e)}")
        
        
        st.sidebar.markdown("---")
        st.sidebar.header("⚙️ Analysis Settings")
        st.session_state.top_n_features = st.sidebar.slider("Top Features", 3, 20, 10)
        st.session_state.max_molecules = st.sidebar.slider("Molecules per Feature", 5, 50, 20)
        
        # Plot configuration
        st.sidebar.header("📊 Plot Configuration")
        st.session_state.plot_config = {
            'font_size': st.sidebar.slider('Base Font Size', 8, 16, 12),
            'title_size': st.sidebar.slider('Title Font Size', 10, 20, 16),
            'label_size': st.sidebar.slider('Label Font Size', 8, 18, 14),
            'tick_size': st.sidebar.slider('Tick Label Size', 6, 14, 12),
            'dpi': st.sidebar.slider('DPI', 100, 500, 300),
            'show_grid': st.sidebar.checkbox('Show Grid', value=True),
            'show_error_bars': st.sidebar.checkbox('Show Error Bars', value=True)
        }

    def _load_demo_data(self):
        """Loads a complete set of demo data from GitHub Releases."""
        st.toast("Loading demo dataset...", icon="⏳")
    
        base_url = "https://github.com/caithmac/shap_analysis/releases/download/Pkl"
    
        dataset_urls = {
        'TYK2': f"{base_url}/TYK2_sorted.csv", 'D2R':  f"{base_url}/composite_d2r.csv",
        'MPRO': f"{base_url}/mpro_sorted.csv", 'USP7': f"{base_url}/USP7_sorted.csv",
    }
        results_urls = {
        'main': f"{base_url}/recall_combined_results__3_.csv",
        'mw':   f"{base_url}/mw_comparison_metrics.csv",
    }
        shap_urls = {
        'TYK2_ucb-exploit-heavy_complete_results.pkl': f"{base_url}/TYK2_ucb-exploit-heavy_complete_results.pkl",
        'TYK2_ucb-explore-heavy_complete_results.pkl': f"{base_url}/TYK2_ucb-explore-heavy_complete_results.pkl",
        'USP7_ucb-exploit-heavy_complete_results.pkl': f"{base_url}/USP7_ucb-exploit-heavy_complete_results.pkl",
        'USP7_ucb-explore-heavy_complete_results.pkl': f"{base_url}/USP7_ucb-explore-heavy_complete_results.pkl",
    }

        self.initialize_session_state() # Reset state

        try:
            for name, url in dataset_urls.items():
                content = download_and_cache_file(url, _file_name=f"{name}_dataset.csv")
                if content: st.session_state.uploaded_datasets[name] = pd.read_csv(io.BytesIO(content))
            st.sidebar.success(f"✓ Loaded {len(st.session_state.uploaded_datasets)} demo datasets.")
        except Exception as e:
            st.sidebar.error(f"Error processing demo datasets: {e}")
            return

        try:
            main_content = download_and_cache_file(results_urls['main'], _file_name="protocol_results.csv")
            if main_content: st.session_state.main_results_df = pd.read_csv(io.BytesIO(main_content))
            mw_content = download_and_cache_file(results_urls['mw'], _file_name="mw_baseline.csv")
            if mw_content: st.session_state.mw_results_df = pd.read_csv(io.BytesIO(mw_content))
            st.sidebar.success("✓ Loaded demo protocol results.")
        except Exception as e:
            st.sidebar.error(f"Error processing protocol results: {e}")

        try:
            for key, url in shap_urls.items():
                content = download_and_cache_file(url, _file_name=f"{key}")
                if content: st.session_state.analysis_data[key] = pickle.load(io.BytesIO(content))
            st.sidebar.success(f"✓ Loaded {len(st.session_state.analysis_data)} demo SHAP analyses.")
        except Exception as e:
            st.sidebar.error(f"Error processing SHAP data: {e}")
    
        self._prepare_molecular_data()
        st.rerun()
    
    def _process_files(self, files, state_key, load_func, by_target=False):
        """Process uploaded files."""
        loaded_count = 0
        for f in files:
            key = f.name
            if by_target:
                # Dynamically check for any known dataset key in the filename
                found_target = None
                for target_key in DATASET_COLORS.keys():
                    if target_key in f.name.upper():
                        found_target = target_key
                        break
                if found_target:
                    key = found_target
            
            if key and key not in st.session_state[state_key]:
                try:
                    st.session_state[state_key][key] = load_func(f)
                    loaded_count += 1
                except Exception as e:
                    st.sidebar.error(f"Failed to load {f.name}: {e}")

        if loaded_count > 0:
            st.sidebar.success(f"Loaded {loaded_count} new file(s).")
    
    def _create_main_interface(self):
        """Create main tabbed interface."""
        tabs = [
            "🏠 Overview",
            "📊 Protocol Performance", 
            "🧬 Chemical Fragments",
            "💊 Drug Design",
            "📈 Feature Evolution",
            "📉 Distribution Analysis",
            "🔬 Molecular Analysis",
            "📄 Publication Figures",
            "🧪 Advanced Analytics"
        ]
        
        tab_objects = st.tabs(tabs)
        
        with tab_objects[0]:
            self._show_overview()

        with tab_objects[1]:
            self._show_protocol_performance()
        
        with tab_objects[2]:
            self._show_chemical_fragments()
        
        with tab_objects[3]:
            self._show_drug_design()
        

        with tab_objects[4]:
            self._show_evolution_analysis()

        
        with tab_objects[5]:
            self._show_distribution_analysis()
        
        with tab_objects[6]:
            self._show_molecular_analysis()
        
        with tab_objects[7]:
            self._show_publication_figures()
        
        with tab_objects[8]:
            self._show_advanced_analytics()
        

    
    def _show_overview(self):
        """Show an overview and user guide for the application, framed for research collaborators."""
        st.markdown('<h2 class="sub-header">Welcome to the Active Learning Analysis Platform</h2>', unsafe_allow_html=True)
    
        st.markdown("""
        This platform provides an interactive interface to explore the comprehensive results of our active learning (AL) benchmarking study. 
        It is designed to bridge the gap between our high-level performance metrics and the deep, mechanistic insights derived from our interpretability analyses.
    
        **The main goals of this platform are to allow you to:**
        - **Explore *our* findings on model interpretability**, by mapping the abstract features from our models back to specific chemical structures.
        - **Interactively compare the performance of the different discovery protocols** we evaluated, to understand their context-dependent efficacy.
        - **Generate the specific, publication-quality figures** used in our manuscript, and explore the underlying data.
        """)
    
        st.markdown("---")
    
        st.markdown("### 🚀 How to Explore Our Results")
    
        col1, col2 = st.columns(2)
    
        with col1:
            st.markdown("#### Option 1: Load Our Pre-analyzed Results (Recommended)")
            st.info("""
            The best way to explore our findings is to load the built-in demo data, which contains the complete results from our study.
        
            1.  Go to the **sidebar on the left**.
            2.  Click the **"🚀 Load Demo Dataset"** button.
        
            This will populate the entire application with our pre-analyzed dataset, allowing you to interactively explore all our findings immediately.
            """)
    
        with col2:
            st.markdown("#### Option 2: Upload Your Own Data (for comparison)")
            st.warning("""
            You can also upload your own data for direct comparison or analysis using our framework.
        
            1.  Open the **sidebar on the left**.
            2.  Use the expanders (`Upload Molecule Datasets`, `Upload Protocol Results`, etc.) to upload your files.
        
            Please ensure your data is in the expected CSV or PKL format.
            """)
        
        st.markdown("---")
    
        st.markdown("### 🗺️ Navigating Our Study's Findings: What Each Tab Shows")
    
        tabs_explanation = {
            "🔬 **Molecular Analysis**": "Start here to explore the chemical data from our study. Visualize the chemical space of the TYK2, USP7, D2R, and MPRO datasets, analyze our model's affinity predictions, and view the top-ranked compounds identified.",
            "📈 **Feature Evolution**": "Track how the importance of key chemical features (identified by our SHAP analysis) changes over the course of our active learning simulations. This helps understand the learning dynamics of our models.",
            "🧬 **Chemical Fragments**": "This is the core interpretation module for our results. It translates the abstract SHAP feature importances from our models into tangible, chemically meaningful fragments. See which substructures our models found to be most important for activity.",
            "📊 **Protocol Performance**": "Rigorously compare the performance of the different discovery strategies we evaluated. View our learning curves and detailed bar charts to see which combination of ML kernel, fingerprint, and protocol worked best for each of our datasets.",
            "📉 **Distribution Analysis**": "Visualize the final performance of the different protocols we tested as distribution plots (ridge plots) to understand the robustness and variance of each method in our study.",
            "💊 **Drug Design**": "Explore computer-aided suggestions for new molecular scaffolds based on the most important chemical fragments identified in our analyses.",
            "🧪 **Advanced Analytics**": "Perform high-level, cross-dataset analyses on our results. Compare feature importance across different targets, analyze feature stability, or identify recurring chemical patterns and SARs from our study.",
            "📄 **Publication Figures**": "Generate the specific, pre-formatted figures used in our manuscript, designed to be high-quality and suitable for direct use in presentations.",
        }
    
        for tab_name, description in tabs_explanation.items():
            with st.expander(tab_name):
                st.write(description)
    
        st.markdown("---")
        
        st.markdown('### 📊 Current Data Status')
    
        # This section provides confirmation of what data is currently loaded.
        if not any([st.session_state.get('molecular_data_df') is not None, st.session_state.main_results_df is not None, st.session_state.analysis_data]):
            st.info("No data is currently loaded. Use the sidebar to load our demo set or upload your own files.")
        else:
            status_col1, status_col2 = st.columns(2)
            with status_col1:
                st.markdown("##### Molecule & Protocol Data")
                if st.session_state.get('molecular_data_df') is not None:
                    df = st.session_state.molecular_data_df
                    source = st.session_state.get('molecular_data_source', 'an available source')
                    st.success(f"**Molecular Data:** {len(df)} molecules loaded from '{source}'.")
                else:
                    st.warning("**Molecular Data:** Not loaded.")
    
                if st.session_state.main_results_df is not None:
                    st.success(f"**Protocol Results:** {len(st.session_state.main_results_df)} records loaded.")
                else:
                    st.warning("**Protocol Results:** Not loaded.")
    
            with status_col2:
                st.markdown("##### SHAP Analysis Data")
                if st.session_state.analysis_data:
                    st.success(f"**SHAP Analyses:** {len(st.session_state.analysis_data)} files loaded.")
                    st.dataframe(
                        pd.DataFrame([{
                            'Analysis Key': key,
                            'Target': data.get('metadata', {}).get('target', 'N/A'),
                            'Protocol': data.get('metadata', {}).get('protocol', 'N/A')
                        } for key, data in st.session_state.analysis_data.items()]),
                        hide_index=True
                )
                else:
                    st.warning("**SHAP Analyses:** Not loaded.")

    
    def _show_evolution_analysis(self):
        """Show feature evolution analysis."""
        st.markdown('<h2 class="sub-header">Feature Evolution Analysis</h2>', unsafe_allow_html=True)
        
        if not st.session_state.analysis_data:
            st.warning("No SHAP analysis data loaded. Please upload analysis files in the sidebar.")
            return
        
        selected_file = st.selectbox("Select analysis file:", list(st.session_state.analysis_data.keys()))
        if not selected_file:
            return
        
        data = st.session_state.analysis_data[selected_file]
        df = data.get('feature_evolution')
        meta = data.get('metadata', {})
        protocol = meta.get('protocol', '')
        
        if df is not None:
            # Get top features
            top_features = df.groupby('feature_index')['importance'].mean().nlargest(
                st.session_state.top_n_features).index
            plot_df = df[df['feature_index'].isin(top_features)]
            
            # Create plot
            fig = px.line(
                plot_df, 
                x='cycle', 
                y='importance', 
                color='feature_index', 
                title=f"Top Feature Evolution: {meta.get('target', '')} - {protocol}",
                labels={'cycle': 'Cycle', 'importance': 'Mean |SHAP|'},
                markers=True
            )
            
            # Add protocol phases
            self._add_protocol_phases(fig, protocol)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature statistics
            with st.expander("Feature Importance Statistics"):
                stats_df = plot_df.groupby('feature_index')['importance'].agg([
                    'mean', 'std', 'min', 'max'
                ]).round(4)
                st.dataframe(stats_df)
    
    def _prepare_molecular_data(self):
            """
            Intelligently sources molecular data for analysis.
            It first looks for a dedicated file (if we add that back later),
            then falls back to combining the 'uploaded_datasets'.
            The result is stored in st.session_state.molecular_data_df.
            """
            # If the data is already prepared, do nothing.
            if 'molecular_data_df' in st.session_state and st.session_state.molecular_data_df is not None:
                return

            # Reset in case of re-runs
            st.session_state.molecular_data_df = None
            st.session_state.molecular_data_source = None

            # Fallback to combining Molecule Datasets
            if st.session_state.uploaded_datasets:
                all_dfs = []
                for dataset_name, df in st.session_state.uploaded_datasets.items():
                    temp_df = df.copy()
                    temp_df['Dataset'] = dataset_name
                    all_dfs.append(temp_df)
                
                if all_dfs:
                    combined_df = pd.concat(all_dfs, ignore_index=True, sort=False)
                    if 'SMILES' in combined_df.columns:
                        st.session_state.molecular_data_df = combined_df
                        st.session_state.molecular_data_source = "combined Molecule Datasets"

    def _add_protocol_phases(self, fig, protocol):
        """Add protocol phase backgrounds to plotly figure."""
        # Define colors
        explore_color = "lightblue"
        exploit_color = "lightcoral" 
        mixed_color = "lightyellow"
        balanced_color = "lightgreen"
        random_color = "lightgray"
        
        # Add initial random phase
        fig.add_vrect(
            x0=-0.5, x1=0.5, 
            fillcolor=random_color, opacity=0.3, 
            layer="below", line_width=0,
            annotation_text="Random",
            annotation_position="top left"
        )
        
        # Add phase backgrounds based on protocol
        if 'exploit-heavy' in protocol:
            fig.add_vrect(x0=0.5, x1=3.5, fillcolor=explore_color, opacity=0.2, 
                         layer="below", line_width=0, annotation_text="Explore Phase")
            fig.add_vrect(x0=3.5, x1=10.5, fillcolor=exploit_color, opacity=0.2, 
                         layer="below", line_width=0, annotation_text="Exploit Phase")
        elif 'explore-heavy' in protocol:
            fig.add_vrect(x0=0.5, x1=7.5, fillcolor=explore_color, opacity=0.2, 
                         layer="below", line_width=0, annotation_text="Explore Phase")
            fig.add_vrect(x0=7.5, x1=10.5, fillcolor=exploit_color, opacity=0.2, 
                         layer="below", line_width=0, annotation_text="Exploit Phase")
        elif 'sandwich' in protocol:
            fig.add_vrect(x0=0.5, x1=2.5, fillcolor=explore_color, opacity=0.2, 
                         layer="below", line_width=0, annotation_text="Explore")
            fig.add_vrect(x0=2.5, x1=8.5, fillcolor=exploit_color, opacity=0.2, 
                         layer="below", line_width=0, annotation_text="Exploit")
            fig.add_vrect(x0=8.5, x1=10.5, fillcolor=explore_color, opacity=0.2, 
                         layer="below", line_width=0, annotation_text="Explore")
        elif 'alternate' in protocol:
            for i in range(1, 11):
                color = explore_color if i % 2 == 1 else exploit_color
                fig.add_vrect(x0=i-0.5, x1=i+0.5, fillcolor=color, opacity=0.15, 
                             layer="below", line_width=0)
        elif 'gradual' in protocol:
            fig.add_vrect(x0=0.5, x1=3.5, fillcolor=explore_color, opacity=0.2, 
                         layer="below", line_width=0, annotation_text="Explore")
            fig.add_vrect(x0=3.5, x1=7.5, fillcolor=mixed_color, opacity=0.2, 
                         layer="below", line_width=0, annotation_text="UCB Mixed")
            fig.add_vrect(x0=7.5, x1=10.5, fillcolor=exploit_color, opacity=0.2, 
                         layer="below", line_width=0, annotation_text="Exploit")
        elif 'balanced' in protocol:
            fig.add_vrect(x0=0.5, x1=10.5, fillcolor=balanced_color, opacity=0.2, 
                         layer="below", line_width=0, annotation_text="Balanced UCB")
        elif 'random' in protocol:
            fig.add_vrect(x0=-0.5, x1=10.5, fillcolor=random_color, opacity=0.2, 
                         layer="below", line_width=0, annotation_text="Random Selection")
    
    def _show_chemical_fragments(self):
        """Show chemical fragment analysis."""
        st.markdown('<h2 class="sub-header">Chemical Fragment Analysis</h2>', unsafe_allow_html=True)
        
        if not st.session_state.analysis_data:
            st.warning("No analysis data available. Please upload SHAP analysis files first.")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_file = st.selectbox(
                "Select analysis:", 
                list(st.session_state.analysis_data.keys()), 
                key="frag_select"
            )
        
        with col2:
            if selected_file:
                meta = st.session_state.analysis_data[selected_file].get('metadata', {})
                st.info(f"**Target:** {meta.get('target', 'Unknown')}\n\n**Protocol:** {meta.get('protocol', 'Unknown')}")
        
        if not selected_file:
            return
        
        # Extract metadata
        meta = st.session_state.analysis_data[selected_file].get('metadata', {})
        target = meta.get('target')
        protocol = meta.get('protocol')
        
        # Validation
        if not target or target not in st.session_state.uploaded_datasets:
            st.error(f"Dataset for target '{target}' not found. Please upload the dataset first.")
            return
        
        # Analysis parameters
        with st.expander("🔧 Analysis Parameters", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_features = st.slider(
                    "Number of top features:", 
                    min_value=5, 
                    max_value=50, 
                    value=st.session_state.top_n_features
                )
            
            with col2:
                max_molecules = st.slider(
                    "Max molecules per feature:", 
                    min_value=10, 
                    max_value=100, 
                    value=st.session_state.max_molecules
                )
            
            with col3:
                importance_threshold = st.slider(
                    "Min importance threshold:", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=0.001,
                    step=0.001,
                    format="%.3f"
                )
        
        # Analysis button
        combo_key = f"{target}_{protocol}"
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            run_analysis = st.button(
                "🧪 Analyze Fragments", 
                key="frag_button", 
                use_container_width=True,
                type="primary"
            )
        
        with col2:
            if combo_key in st.session_state.fragment_mappings:
                st.success("✅ Analysis cached and ready to view")
                if st.button("🔄 Re-run Analysis", key="rerun_frag", use_container_width=True):
                    run_analysis = True
        
        # Run analysis
        if run_analysis:
            try:
                with st.spinner(f"🔍 Mapping fragments for {target}-{protocol}..."):
                    mapper = ChemicalFragmentMapper(
                        st.session_state.analysis_data[selected_file], 
                        st.session_state.uploaded_datasets[target], 
                        smiles_to_ecfp8
                    )
                    
                    fragments = mapper.extract_fragments_for_features(
                        target, 
                        protocol, 
                        n_features, 
                        max_molecules
                    )
                    
                    st.session_state.fragment_mappings[combo_key] = (mapper, fragments)
                    
            except Exception as e:
                st.error(f"Error during fragment analysis: {str(e)}")
                return
        
        # Display results
        if combo_key in st.session_state.fragment_mappings:
            mapper, fragments = st.session_state.fragment_mappings[combo_key]
            
            if not fragments:
                st.warning("No fragments found meeting the specified criteria.")
                return
            
            st.markdown("---")
            st.markdown("### 📊 Analysis Results")
            
            # Summary statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Features Found", len(fragments))
            
            with col2:
                total_molecules = sum(len(data['molecules_with_feature']) for data in fragments.values())
                st.metric("Total Molecules", total_molecules)
            
            # Display features
            for feature_idx, data in fragments.items():
                header = f"**Feature {feature_idx}** | Rank #{data['rank']} | Importance: {data['average_importance']:.4f}"
                
                with st.expander(header):
                    self._render_feature_detail_tabs(selected_file, feature_idx, data, mapper)
    
    def _render_feature_detail_tabs(self, selected_file, feature_idx, data, mapper):
        """Render detailed tabs for a feature."""
        tab1, tab2, tab3, tab4 = st.tabs(["Fragment Overview", "Fragment Details", "SHAP Contribution", "Feature Impact"])
        
        with tab1:
            col1, col2 = st.columns([0.4, 0.6])
            with col1:
                st.markdown("**All Common Fragments**")
                st.info("Gallery of all frequent fragment structures for this feature.", icon="🧩")
                comprehensive_gallery = mapper.get_comprehensive_fragment_gallery(data['fragments'])
                if isinstance(comprehensive_gallery, plt.Figure):
                    st.pyplot(comprehensive_gallery)
                elif comprehensive_gallery:
                    st.image(comprehensive_gallery)
                else:
                    st.warning("No common fragments identified.")
                
                st.markdown("**Affinity Distribution**")
                affinity_fig = mapper.get_affinity_plot_fig(data)
                if affinity_fig:
                    st.pyplot(affinity_fig)
            
            with col2:
                st.markdown("**Highlighted High-Affinity Molecules**")
                st.info("Examples of where fragments appear in high-affinity molecules.", icon="🎯")
                for highlight_info in data['fragments']['molecule_highlights'][:3]:
                    mol_img = create_molecule_aura_image(
                        highlight_info['mol'], 
                        highlight_info['highlight_atoms'], 
                        highlight_info['highlight_bonds']
                    )
                    if mol_img:
                        st.image(mol_img, caption=f"Affinity: {highlight_info['parent_info']['affinity']:.2f}")
        
        with tab2:
            st.markdown(f"### Detailed View: Most Common Fragment")
            st.info("This shows the most common fragment isolated and examples of where it appears.", icon="🔬")
            isolated_img, parent_images = mapper.get_integrated_fragment_display(data['fragments'])
            
            if isolated_img and parent_images:
                st.markdown("#### Isolated Fragment Structure")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(isolated_img, caption=f"Most Common Fragment: {data['fragments']['most_common'][0][0]}")
                
                st.markdown("#### This Fragment in High-Affinity Molecules")
                st.markdown("*The same fragment structure highlighted in its parent molecules:*")
                cols = st.columns(len(parent_images) if parent_images else 1)
                for i, (col, p_info) in enumerate(zip(cols, parent_images)):
                    with col:
                        st.image(p_info['image'], caption=f"Affinity: {p_info['affinity']:.2f}", use_column_width=True)
            else:
                st.warning("Could not generate detailed fragment analysis.")
        
        # SHAP and Impact tabs would require cycle_results data
        with tab3:
            st.info("SHAP contribution analysis requires cycle results data.", icon="📈")
        
        with tab4:
            st.info("Feature impact analysis requires cycle results data.", icon="🎯")
    
    def _show_drug_design(self):
        """Show molecular design templates."""
        st.markdown('<h2 class="sub-header">Molecular Design Templates</h2>', unsafe_allow_html=True)
        st.info("Generates synthesizable molecular scaffolds based on important chemical fragments.")
        
        if not st.session_state.uploaded_datasets:
            st.warning("No datasets loaded. Please upload molecule datasets first.")
            return
        
        target_to_design = st.selectbox("Select target:", list(st.session_state.uploaded_datasets.keys()), key="design_select")
        
        if st.button("Generate Design Templates", key="design_button", use_container_width=True):
            generator = MolecularDesignTemplateGenerator()
            scaffolds = generator.generate_scaffolds(target_to_design)
            
            if not scaffolds:
                st.warning("No design templates available for this target.")
                return
            
            st.markdown(f"### Proposed Scaffolds for {target_to_design}")
            st.image(generator.get_scaffold_grid_image(scaffolds))
            
            scaffold_data = []
            for n, d in scaffolds.items():
                props = generator.calculate_properties(d['smiles'])
                scaffold_data.append({
                    'Scaffold': d['name'], 
                    'SMILES': d['smiles'], 
                    **props
                })
            
            st.dataframe(pd.DataFrame(scaffold_data).round(2), use_container_width=True)
    
    def _show_protocol_performance(self):
        """Show protocol performance visualizations."""
        st.markdown('<h2 class="sub-header">Protocol Performance Analysis</h2>', unsafe_allow_html=True)
        
        if st.session_state.main_results_df is None:
            st.warning("No protocol results data loaded. Please upload main results CSV in the sidebar.")
            return
        
        df = st.session_state.main_results_df
        
        plot_type = st.selectbox(
            "Select analysis type:",
            ["Protocol Comparison", "Dataset Performance", "Kernel Performance", "Fingerprint Performance"]
        )
        
        if plot_type == "Protocol Comparison":
            protocols = sorted(df['Protocol'].unique())
            selected_protocol = st.selectbox("Select Protocol", protocols)
            
            kernels = sorted(df['Kernel'].unique())
            colors = create_color_picker("Kernel", len(kernels))
            
            if st.button('Generate Plot'):
                fig = create_protocol_comparison_plot(
                    df, selected_protocol,
                    sorted(df['Dataset'].unique()),
                    kernels,
                    sorted(df['Fingerprint'].unique()),
                    colors, st.session_state.plot_config
                )
                st.pyplot(fig)
                
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches='tight')
                buf.seek(0)
                st.download_button("Download Plot", buf.getvalue(), f"protocol_comparison_{selected_protocol}.png", "image/png")
        
        elif plot_type == "Dataset Performance":
            datasets = sorted(df['Dataset'].unique())
            selected_dataset = st.selectbox("Select Dataset", datasets)
            
            num_colors = len(df['Kernel'].unique()) * len(df['Fingerprint'].unique())
            colors = create_color_picker("Kernel-Fingerprint", num_colors)
            
            if st.button('Generate Plot'):
                mw_df = st.session_state.mw_results_df if st.session_state.mw_results_df is not None else pd.DataFrame()
                
                fig = create_dataset_performance_plot(
                    df=df,
                    mw_df=mw_df,
                    dataset=selected_dataset,
                    kernels=sorted(df['Kernel'].unique()),
                    fingerprints=sorted(df['Fingerprint'].unique()),
                    colors=colors,
                    plot_config=st.session_state.plot_config
                )
                st.pyplot(fig)
                
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches='tight')
                buf.seek(0)
                st.download_button("Download Plot", buf.getvalue(), f"dataset_performance_{selected_dataset}.png", "image/png")
        
        elif plot_type == "Kernel Performance":
            kernels = sorted(df['Kernel'].unique())
            selected_kernel = st.selectbox("Select Kernel", kernels)
            
            fingerprints = sorted(df['Fingerprint'].unique())
            colors = create_color_picker("Fingerprint", len(fingerprints))
            
            if st.button('Generate Plot'):
                fig = create_kernel_performance_plot(
                    df=df,
                    kernel=selected_kernel,
                    datasets=sorted(df['Dataset'].unique()),
                    protocols=sorted(df['Protocol'].unique()),
                    fingerprints=fingerprints,
                    colors=colors,
                    plot_config=st.session_state.plot_config
                )
                st.pyplot(fig)

                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches='tight')
                buf.seek(0)
                st.download_button("Download Plot", buf.getvalue(), f"kernel_performance_{selected_kernel}.png", "image/png")
        
        elif plot_type == "Fingerprint Performance":
            fingerprints = sorted(df['Fingerprint'].unique())
            selected_fingerprint = st.selectbox("Select Fingerprint", fingerprints)

            kernels = sorted(df['Kernel'].unique())
            colors = create_color_picker("Kernel", len(kernels))
            
            if st.button('Generate Plot'):
                fig = create_fingerprint_performance_plot(
                    df=df,
                    fingerprint=selected_fingerprint,
                    datasets=sorted(df['Dataset'].unique()),
                    protocols=sorted(df['Protocol'].unique()),
                    kernels=kernels,
                    colors=colors,
                    plot_config=st.session_state.plot_config
                )
                st.pyplot(fig)

                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches='tight')
                buf.seek(0)
                st.download_button("Download Plot", buf.getvalue(), f"fingerprint_performance_{selected_fingerprint}.png", "image/png")
    
    def _show_distribution_analysis(self):
        """Show distribution analysis (ridge plots)."""
        st.markdown('<h2 class="sub-header">Distribution Analysis</h2>', unsafe_allow_html=True)
        
        if st.session_state.main_results_df is None:
            st.warning("No protocol results data loaded. Please upload main results CSV.")
            return
        
        df = st.session_state.main_results_df
        mw_df = st.session_state.mw_results_df if st.session_state.mw_results_df is not None else pd.DataFrame()
        
        datasets = sorted(df['Dataset'].unique())
        selected_dataset = st.selectbox("Select Dataset", datasets)
        
        if st.button('Generate Ridge Plot'):
            with st.spinner('Generating ridge plot...'):
                try:
                    fig = create_ridge_plot_v3(df, mw_df, selected_dataset)
                    st.pyplot(fig)
                    
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight', 
                              dpi=st.session_state.plot_config['dpi'])
                    buf.seek(0)
                    st.download_button(
                        label="Download Plot",
                        data=buf,
                        file_name=f"ridge_plot_{selected_dataset}.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.error(f"Error generating plot: {str(e)}")
    
    def _show_molecular_analysis(self):
        """Show molecular analysis visualizations."""
        st.markdown('<h2 class="sub-header">Molecular Analysis</h2>', unsafe_allow_html=True)

        if st.session_state.get('molecular_data_df') is None:
            st.warning("No molecular data available. Please upload 'Molecule Datasets' in the sidebar.")
            return

        master_df = st.session_state.molecular_data_df
        
        available_datasets = sorted(master_df['Dataset'].unique())
        
        if len(available_datasets) > 1:
            st.info(f"Found {len(available_datasets)} datasets. Please select which one(s) to analyze.", icon="ℹ️")
            selected_datasets = st.multiselect(
                "Select Datasets to Analyze",
                options=available_datasets,
                default=[available_datasets[0]],
                key="molecular_analysis_dataset_selector"
            )
        else:
            selected_datasets = available_datasets

        if not selected_datasets:
            st.warning("Please select at least one dataset to continue.")
            return

        df_to_analyze = master_df[master_df['Dataset'].isin(selected_datasets)]
        
        st.success(f"Analyzing **{len(df_to_analyze)}** molecules from **{len(selected_datasets)}** selected dataset(s).", icon="🔬")

        st.sidebar.markdown("---")
        st.sidebar.header("🔬 Molecular Analysis Filters")

        filtered_df = df_to_analyze.copy()

        if 'affinity' in filtered_df.columns:
            min_aff, max_aff = float(filtered_df['affinity'].min()), float(filtered_df['affinity'].max())
            if min_aff < max_aff:
                aff_range = st.sidebar.slider(
                    "Filter by Affinity",
                    min_value=min_aff,
                    max_value=max_aff,
                    value=(min_aff, max_aff)
                )
                filtered_df = filtered_df[filtered_df['affinity'].between(aff_range[0], aff_range[1])]

        analysis_type = st.selectbox(
            "Select analysis type:",
            ["Affinity Predictions", "Chemical Space", "Top Compounds"]
        )
        
        if analysis_type == "Affinity Predictions":
            pred_cols = [col for col in filtered_df.columns if col.startswith('pred_')]
            if not pred_cols:
                st.warning("No prediction columns found in the selected molecular data.")
            else:
                cycles = sorted([int(col.split('_')[1]) for col in pred_cols if col.split('_')[1].isdigit()])
                selected_cycle = st.selectbox("Select Prediction Cycle", cycles)
                if st.button('Generate Affinity Plot', use_container_width=True):
                    fig = plot_affinity_predictions(filtered_df, selected_cycle)
                    if fig:
                        st.pyplot(fig)
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', bbox_inches='tight')
                        st.download_button(label="Download Plot", data=buf, file_name=f"affinity_predictions_cycle_{selected_cycle}.png", mime="image/png")

        elif analysis_type == "Chemical Space":
            st.info("Visualize the chemical landscape of your molecules. Hover over points for details.", icon="🗺️")

            color_options = []
            if 'Dataset' in filtered_df.columns and len(filtered_df['Dataset'].unique()) > 1:
                color_options.append('Dataset')
            if 'affinity' in filtered_df.columns:
                color_options.append('affinity')
            color_options.extend([col for col in filtered_df.columns if col.startswith('pred_')])
            if not color_options:
                st.warning("No columns available for coloring.")
                return

            col1, col2, col3 = st.columns(3)
            with col1:
                color_by = st.selectbox("Color By", color_options)
            with col2:
                methods = ['PCA', 't-SNE']
                if HAS_UMAP:
                    methods.append('UMAP')
                method = st.selectbox("Reduction Method", methods)
            with col3:
                fp_type = st.selectbox("Fingerprint Type", ['ECFP', 'MACCS'])

            if st.button('Generate Chemical Space Plot', use_container_width=True):
                fig, plot_df_result = visualize_chemical_space(filtered_df, color_by, method, fp_type)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    with st.expander("Plot Data Summary"):
                        if plot_df_result is not None:
                            st.write(f"Total molecules visualized: {len(plot_df_result)}")
                            if color_by == 'Dataset':
                                st.dataframe(plot_df_result.groupby('Dataset').size().reset_index(name='count'))
                            else:
                                st.dataframe(plot_df_result.drop(columns=['SMILES'], errors='ignore').describe())
                    csv = plot_df_result.to_csv(index=False).encode('utf-8')
                    st.download_button(label="Download Plot Data (CSV)", data=csv, file_name=f"chemical_space_{method}_{color_by}.csv", mime="text/csv")

        elif analysis_type == "Top Compounds":
            sort_options = []
            if 'affinity' in filtered_df.columns:
                sort_options.append('affinity')
            sort_options.extend([col for col in filtered_df.columns if col.startswith('pred_')])
            if not sort_options:
                st.warning("No columns available for sorting.")
                return
            
            sort_by = st.selectbox("Sort By", sort_options, key="top_compounds_sort")
            n_compounds = st.slider("Number of Compounds", 1, 20, 5, key="top_compounds_slider")
            ascending = st.checkbox("Ascending Order (show worst)", value=False, key="top_compounds_checkbox")

            if st.button('Show Top Compounds', use_container_width=True):
                img, top_df = visualize_top_compounds(filtered_df, n=n_compounds, sort_by=sort_by, ascending=ascending)
                if img and top_df is not None:
                    st.image(img)
                    st.subheader(f"{'Top' if not ascending else 'Bottom'} {n_compounds} Compounds by {sort_by}")
                    st.dataframe(top_df.round(3))
                    csv = top_df.to_csv(index=False).encode('utf-8')
                    st.download_button(label="Download Data as CSV", data=csv, file_name=f"top_compounds_{sort_by}.csv", mime="text/csv")
                else:
                    st.warning("Could not generate top compounds view.")
    
    def _show_publication_figures(self):
        """Show publication-ready figures."""
        st.markdown('<h2 class="sub-header">Publication-Ready Figures</h2>', unsafe_allow_html=True)
        st.info("Generates high-quality figures summarizing key findings.")
        
        if not st.session_state.fragment_mappings:
            st.warning("Run 'Chemical Fragment Analysis' first to generate publication figures.")
            return
        
        combo_key = st.selectbox("Select analysis for figure:", list(st.session_state.fragment_mappings.keys()), key="pub_fig_select")
        
        if st.button("Generate Main Figure", key="pub_fig_button", use_container_width=True):
            target, protocol = combo_key.split('_', 1)
            generator = PublicationFigureGenerator(st.session_state.fragment_mappings, st.session_state.uploaded_datasets)
            fig = generator.create_main_figure(target, protocol)
            
            if fig:
                st.pyplot(fig)
                
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches='tight')
                buf.seek(0)
                st.download_button("Download Figure", buf.getvalue(), f"{combo_key}_key_features.png", "image/png")
    
    def _show_advanced_analytics(self):
        """Show advanced analytics options."""
        st.markdown('<h2 class="sub-header">Advanced Analytics & Insights</h2>', unsafe_allow_html=True)
        st.write("This section provides high-level comparisons across all loaded experimental conditions.")
        
        analysis_type = st.selectbox(
            "Choose an analysis:", 
            [
                "Cross-Target Feature Heatmap", 
                "Feature Stability Analysis", 
                "Chemical Pattern Recognition", 
                "Structure-Activity Relationship (SAR)",
                "Similarity Distribution Analysis",
                "Molecular Property Analysis",
                "Performance Summary Analysis"
            ]
        )
        
        if analysis_type == "Cross-Target Feature Heatmap":
            self._create_cross_target_heatmap()
        elif analysis_type == "Feature Stability Analysis":
            self._analyze_feature_stability()
        elif analysis_type == "Chemical Pattern Recognition":
            self._identify_chemical_patterns()
        elif analysis_type == "Structure-Activity Relationship (SAR)":
            self._perform_sar_analysis()
        elif analysis_type == "Similarity Distribution Analysis":
            self._show_similarity_analysis(key_prefix="adv_sim")
        elif analysis_type == "Molecular Property Analysis":
            self._show_property_analysis(key_prefix="adv_prop")
        elif analysis_type == "Performance Summary Analysis":
            self._show_performance_summary(key_prefix="adv_perf")
    
    def _show_publication_gallery(self):
        """Show publication gallery with pre-formatted figures."""
        st.markdown('<h2 class="sub-header">Publication Gallery</h2>', unsafe_allow_html=True)
        st.info("Access pre-formatted publication figures with one click.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Generate Figure 1: Similarity Distributions", use_container_width=True):
                if st.session_state.uploaded_datasets:
                    self._show_similarity_analysis(key_prefix="pub_gallery_sim")
        
        with col2:
            if st.button("Generate Figure 2: Property Distributions", use_container_width=True):
                if st.session_state.uploaded_datasets:
                    self._show_property_analysis(key_prefix="pub_gallery_prop")
        
        with col3:
            if st.button("Generate Figure 3: Performance Summary", use_container_width=True):
                if st.session_state.main_results_df is not None:
                    self._show_performance_summary(key_prefix="pub_gallery_perf")
        
        st.markdown("---")
        st.markdown("### Additional Figures")
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            if st.button("Generate Figure 4: Performance Heatmap", use_container_width=True):
                if st.session_state.main_results_df is not None:
                    # Heatmap is part of the performance summary function
                    self._show_performance_summary(key_prefix="pub_gallery_heatmap")
        
        with col5:
            if st.button("Generate Chemical Space Figure", use_container_width=True):
                if st.session_state.get('molecular_data_df') is not None:
                    st.info("Please navigate to the 'Molecular Analysis' tab to generate this figure.")
                else:
                    st.warning("No molecular data loaded. Please upload 'Molecule Datasets'.")
        
        with col6:
            if st.button("Generate Feature Evolution", use_container_width=True):
                if st.session_state.analysis_data:
                    st.info("Please navigate to the 'Feature Evolution' tab to generate this figure.")

    def _show_similarity_analysis(self, key_prefix="sim"):
        """Show molecular similarity distribution analysis."""
        st.info("Analyze the chemical diversity of your datasets using different molecular representations.", icon="🔬")
        
        if not st.session_state.uploaded_datasets:
            st.warning("Please upload molecule datasets first.")
            return
        
        available_datasets = list(st.session_state.uploaded_datasets.keys())
        selected_datasets = st.multiselect(
            "Select datasets to analyze:",
            available_datasets,
            default=available_datasets[:min(4, len(available_datasets))],
            key=f"{key_prefix}_datasets_multiselect"
        )

        if not selected_datasets:
            st.warning("Please select at least one dataset.")
            return
        
        col1, col2 = st.columns(2)
        with col1:
            sample_size = st.slider(
                "Sample size per dataset:",
                min_value=100,
                max_value=5000,
                value=1500,
                step=100,
                help="Number of molecules to sample for similarity calculation",
                key=f"{key_prefix}_sample_size_slider"
            )
        
        with col2:
            include_chemberta = st.checkbox(
                "Include ChemBERTa embeddings",
                value=False,
                help="ChemBERTa analysis requires GPU and may be slow",
                key=f"{key_prefix}_chemberta_checkbox"
            )
        
        if st.button("Generate Similarity Analysis", key=f"{key_prefix}_generate_btn", use_container_width=True):
            with st.spinner("Analyzing molecular similarities..."):
                all_compounds_list = []
                for dataset_name in selected_datasets:
                    df = st.session_state.uploaded_datasets[dataset_name]
                    df['Dataset'] = dataset_name
                    all_compounds_list.append(df)
                
                all_compounds_df = pd.concat(all_compounds_list, ignore_index=True)
                
                fig_path = 'similarity_distributions.png'
                create_similarity_distribution_plot(
                    all_compounds_df,
                    selected_datasets,
                    filename=fig_path,
                    sample_size=sample_size
                )
                
                st.image(fig_path)
                
                with open(fig_path, 'rb') as f:
                    st.download_button(
                        label="Download Similarity Analysis",
                        data=f.read(),
                        file_name="similarity_distributions.png",
                        mime="image/png"
                    )
    
    def _show_property_analysis(self, key_prefix="prop"):
        """Show molecular property distribution analysis."""
        st.info("Analyze the distribution of key molecular properties across your datasets.", icon="💊")
        
        if not st.session_state.uploaded_datasets:
            st.warning("Please upload molecule datasets first.")
            return
        
        available_datasets = list(st.session_state.uploaded_datasets.keys())
        selected_datasets = st.multiselect(
            "Select datasets to analyze:",
            available_datasets,
            default=available_datasets[:min(4, len(available_datasets))],
            key=f"{key_prefix}_datasets_multiselect"
        )
        
        if not selected_datasets:
            st.warning("Please select at least one dataset.")
            return
        
        if st.button("Generate Property Analysis", key=f"{key_prefix}_generate_btn", use_container_width=True):
            with st.spinner("Calculating molecular properties..."):
                all_compounds_list = []
                for dataset_name in selected_datasets:
                    df = st.session_state.uploaded_datasets[dataset_name]
                    df['Dataset'] = dataset_name
                    all_compounds_list.append(df)
                
                all_compounds_df = pd.concat(all_compounds_list, ignore_index=True)
                
                props_df = calculate_molecular_properties(all_compounds_df)
                
                st.markdown("### Property Statistics")
                for dataset in selected_datasets:
                    dataset_props = props_df[props_df['Dataset'] == dataset]
                    if not dataset_props.empty:
                        st.markdown(f"**{dataset}**")
                        st.write(dataset_props[['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds']].describe())
                
                fig_path = 'property_distributions.png'
                create_property_distribution_boxplots(
                    props_df,
                    selected_datasets,
                    filename=fig_path
                )
                
                st.image(fig_path)
                
                with open(fig_path, 'rb') as f:
                    st.download_button(
                        label="Download Property Analysis",
                        data=f.read(),
                        file_name="property_distributions.png",
                        mime="image/png"
                    )
    
    def _show_performance_summary(self, key_prefix="perf_summary"):
        """Show comprehensive performance summary analysis."""
        st.info("Generate publication-ready performance summary figures.", icon="📊")
        
        if st.session_state.main_results_df is None:
            st.warning("Please upload protocol results data first.")
            return
        
        df = st.session_state.main_results_df
        
        available_datasets = sorted(df['Dataset'].unique())
        selected_datasets = st.multiselect(
            "Select datasets to include:",
            available_datasets,
            default=available_datasets[:min(4, len(available_datasets))],
            key=f"{key_prefix}_datasets_multiselect"
        )
        
        if not selected_datasets:
            st.warning("Please select at least one dataset.")
            return
        
        col1, col2 = st.columns(2)
        with col1:
            success_threshold = st.slider(
                "Success threshold for recall:",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Threshold for counting successful methods",
                key=f"{key_prefix}_threshold_slider"
            )
        
        with col2:
            create_heatmap = st.checkbox(
                "Also create performance heatmap",
                value=True,
                help="Generate comprehensive heatmap of all results",
                key=f"{key_prefix}_heatmap_checkbox"
            )
        
        if st.button("Generate Performance Summary", key=f"{key_prefix}_generate_btn", use_container_width=True):
            with st.spinner("Creating performance summary..."):
                summary_path = 'performance_summary.png'
                create_performance_summary_figure(
                    df,
                    selected_datasets,
                    filename=summary_path
                )
                
                st.markdown("### Performance Summary")
                st.image(summary_path)
                
                with open(summary_path, 'rb') as f:
                    st.download_button(
                        label="Download Performance Summary",
                        data=f.read(),
                        file_name="performance_summary.png",
                        mime="image/png"
                    )
                
                if create_heatmap:
                    heatmap_path = 'performance_heatmap.png'
                    create_full_performance_heatmap(
                        df,
                        filename=heatmap_path
                    )
                    
                    st.markdown("### Performance Heatmap")
                    st.image(heatmap_path)
                    
                    with open(heatmap_path, 'rb') as f:
                        st.download_button(
                            label="Download Performance Heatmap",
                            data=f.read(),
                            file_name="performance_heatmap.png",
                            mime="image/png"
                        )
    
    def _create_cross_target_heatmap(self):
        """Create cross-target feature importance heatmap."""
        st.info("This heatmap compares the normalized importance of key features across all loaded experiments.", icon="🗺️")
        
        if len(st.session_state.analysis_data) < 2:
            st.warning("Upload at least two analysis files for comparison.")
            return
        
        if st.button("Generate Heatmap", key="heatmap_btn", use_container_width=True):
            with st.spinner("Generating cross-target heatmap..."):
                all_importances = []
                
                for filename, data in st.session_state.analysis_data.items():
                    meta = data.get('metadata', {})
                    df = data.get('feature_evolution')
                    
                    if df is not None and meta:
                        combo_key = f"{meta.get('target', 'T')}_{meta.get('protocol', 'P')}"
                        mean_importance = df.groupby('feature_index')['importance'].mean().reset_index()
                        mean_importance['combination'] = combo_key
                        all_importances.append(mean_importance)
                
                if not all_importances:
                    st.error("Could not extract importance data.")
                    return
                
                full_df = pd.concat(all_importances, ignore_index=True)
                top_features = set(full_df.groupby('combination').apply(
                    lambda x: x.nlargest(20, 'importance')['feature_index']).explode())
                
                pivot_df = full_df[full_df['feature_index'].isin(top_features)].pivot_table(
                    index='feature_index', columns='combination', values='importance', fill_value=0)
                
                norm_pivot_df = pivot_df.apply(lambda x: (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) > 0 else 0, axis=1)
                
                if norm_pivot_df.empty:
                    st.warning("Not enough data to generate a heatmap.")
                    return
                    
                st.markdown("### Clustered Heatmap of Feature Importance")
                clustergrid = sns.clustermap(norm_pivot_df, cmap='viridis', 
                                           figsize=(max(10, len(pivot_df.columns) * 0.8), max(10, len(pivot_df.index) * 0.3)), 
                                           dendrogram_ratio=(0.1, 0.2), yticklabels=True)
                
                plt.setp(clustergrid.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
                st.pyplot(clustergrid.fig)
                
                buf = io.BytesIO()
                clustergrid.savefig(buf, format="png", bbox_inches='tight')
                st.download_button("Download Heatmap", buf.getvalue(), "cross_target_heatmap.png", "image/png")
    
    def _analyze_feature_stability(self):
        """Analyze feature stability across cycles."""
        st.info("This analysis reveals which features are consistently important across active learning cycles.", icon="⚓")
        
        selected_file = st.selectbox("Select an experiment to analyze for stability:", 
                                    list(st.session_state.analysis_data.keys()), key="stability_select")
        if not selected_file:
            return
        
        data = st.session_state.analysis_data[selected_file]
        df = data.get('feature_evolution')
        
        if df is None:
            st.warning("Selected file does not contain feature evolution data.")
            return
        
        with st.spinner("Calculating feature stability..."):
            stability_df = df.groupby('feature_index')['importance'].agg(['mean', 'std']).reset_index()
            stability_df['cv'] = (stability_df['std'] / stability_df['mean']).fillna(0)
            
            st.markdown("### Stability vs. Importance Plot")
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.scatterplot(data=stability_df, x='mean', y='cv', ax=ax, alpha=0.6, s=50)
            ax.set_xlabel("Mean Importance (Impact)")
            ax.set_ylabel("Coefficient of Variation (Stability)")
            ax.set_title("Feature Stability Analysis")
            st.pyplot(fig)
            
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight')
            st.download_button("Download Stability Analysis", buf.getvalue(), "feature_stability.png", "image/png")
    
    def _identify_chemical_patterns(self):
        """Identify recurring chemical patterns across datasets."""
        st.info("Analyze molecular scaffolds and fragments that appear across multiple datasets.", icon="🔍")
        
        if not st.session_state.uploaded_datasets:
            st.warning("Upload molecule datasets first.")
            return
        
        if st.button("Identify Chemical Patterns", key="pattern_btn", use_container_width=True):
            with st.spinner("Analyzing chemical patterns..."):
                try:
                    all_compounds = []
                    for dataset_name, df in st.session_state.uploaded_datasets.items():
                        df['Dataset'] = dataset_name
                        all_compounds.append(df)
                    
                    combined_df = pd.concat(all_compounds, ignore_index=True)
                    
                    scaffolds = []
                    for smiles in combined_df['SMILES']:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                            scaffolds.append(Chem.MolToSmiles(scaffold))
                        else:
                            scaffolds.append(None)
                    
                    combined_df['Scaffold'] = scaffolds
                    scaffold_counts = combined_df['Scaffold'].value_counts().head(20)
                    
                    st.markdown("### Most Common Scaffolds")
                    st.write(scaffold_counts)
                    
                    mols = []
                    legends = []
                    for scaffold, count in scaffold_counts.head(5).items():
                        try:
                            mol = Chem.MolFromSmiles(scaffold)
                            if mol:
                                mols.append(mol)
                                legends.append(f"Count: {count}")
                        except:
                            pass
                    
                    if mols:
                        img = Draw.MolsToGridImage(mols, molsPerRow=min(3, len(mols)), subImgSize=(300, 300), 
                                                legends=legends, useSVG=False)
                        st.image(img)
                    
                except Exception as e:
                    st.error(f"Error in chemical pattern analysis: {str(e)}")
    
    def _perform_sar_analysis(self):
        """Perform structure-activity relationship analysis."""
        st.info("Analyze the relationship between molecular structure and binding affinity.", icon="📐")
        
        if not st.session_state.uploaded_datasets:
            st.warning("Upload molecule datasets first.")
            return
        
        selected_target = st.selectbox("Select target for SAR analysis:", 
                                      list(st.session_state.uploaded_datasets.keys()))
        if not selected_target:
            return
        
        if st.button("Perform SAR Analysis", key="sar_btn", use_container_width=True):
            with st.spinner("Analyzing structure-activity relationships..."):
                try:
                    df = st.session_state.uploaded_datasets[selected_target]
                    analyzer = SARAnalyzer(df)
                    
                    st.markdown("### R-Group Analysis")
                    core, r_group_df = analyzer.analyze_r_groups()
                    st.write(f"Most common scaffold: {Chem.MolToSmiles(core)}")
                    st.dataframe(r_group_df.head(), use_container_width=True)
                    
                    st.markdown("### Activity Cliffs")
                    cliffs = analyzer.analyze_activity_cliffs()
                    if cliffs:
                        st.write(f"Found {len(cliffs)} activity cliffs")
                        for i, cliff in enumerate(cliffs[:3]):
                            st.write(f"**Cliff {i+1}**: ΔAffinity = {cliff['delta']:.2f}, Similarity = {cliff['similarity']:.2f}")
                            img = Draw.MolsToGridImage([cliff['mol1'], cliff['mol2']], molsPerRow=2, 
                                                     legends=[f"Affinity: {cliff['act1']:.2f}", 
                                                              f"Affinity: {cliff['act2']:.2f}"])
                            st.image(img)
                    else:
                        st.warning("No significant activity cliffs found")
                    
                except Exception as e:
                    st.error(f"Error in SAR analysis: {str(e)}")

# ==============================================================================
# 7. APPLICATION ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    app = UnifiedDrugDiscoveryApp()
    app.run()

