# SHAP Analysis Application

A comprehensive Streamlit-based platform for analyzing active learning experiments in molecular property prediction, with integrated SHAP-based interpretability and chemical fragment mapping.

---

## 📋 Overview

This application provides an interactive interface for exploring active learning (AL) results in drug discovery, combining:

- **SHAP-guided interpretability**: Map abstract ML features to concrete chemical substructures
- **Protocol performance analysis**: Compare different AL strategies across multiple datasets
- **Chemical fragment mapping**: Visualize which molecular fragments drive model predictions
- **Molecular design templates**: Generate synthesizable scaffolds based on important features
- **Publication-quality visualizations**: Export manuscript-ready figures

---

## 🚀 Quick Start



### Running the Application

```bash
# Launch the Streamlit app
streamlit run apps/shap_app/shapey.py

# Or use the launcher script
python apps/shap_app/launch_shap.py
```

---

## 📊 Loading Demo Data

The easiest way to explore the platform is to load the pre-analyzed demo dataset:

1. Click the **"🚀 Load Demo Dataset"** button in the sidebar
2. The app will download example data from GitHub releases
3. All tabs will become populated with interactive analyses

### Demo Data Includes:
- **4 protein targets**: TYK2, USP7, D2R, MPRO
- **Multiple AL protocols**: UCB-explore-heavy, UCB-exploit-heavy, balanced, random
- **SHAP analysis results**: Pre-computed feature importance for key protocols
- **Protocol performance metrics**: Complete benchmarking results

---

## 📁 Data Format Requirements

### 1. Molecule Datasets (CSV)

Required columns:
- `SMILES`: Molecular structure in SMILES format
- `affinity`: Binding affinity values (continuous)

Optional columns:
- `Dataset`: Dataset identifier
- `pred_{cycle}`: Model predictions at each AL cycle

Example:
```csv
SMILES,affinity,Dataset
CCO,5.2,TYK2
c1ccccc1,4.8,TYK2
```

### 2. Protocol Results (CSV)

Required columns:
- `Dataset`: Target protein name
- `Protocol`: AL protocol identifier
- `Kernel`: GP kernel type
- `Fingerprint`: Molecular representation (ecfp/maccs/chemberta)
- `Cycle`: AL iteration number
- `Compounds acquired`: Number of molecules acquired
- `Recall (2%)`: Performance metric
- `Seed`: Random seed

### 3. SHAP Analysis Files (PKL)

Pickle files containing:
```python
{
    'metadata': {
        'target': 'TYK2',
        'protocol': 'ucb-exploit-heavy',
        'fingerprint': 'ecfp',
        'kernel': 'rbf'
    },
    'feature_evolution': pd.DataFrame({
        'cycle': [0, 1, 2, ...],
        'feature_index': [42, 105, ...],
        'importance': [0.15, 0.12, ...]
    })
}
```

---

## 🗺️ Application Structure

### Main Tabs

#### 🏠 **Overview**
- Application guide and data status
- Quick-start instructions
- Data loading confirmation

#### 📊 **Protocol Performance**
- Compare AL strategies across datasets
- Learning curves with error bands
- Bar charts of final performance
- Customizable plot parameters

#### 🧬 **Chemical Fragments**
- Map SHAP features to molecular substructures
- Visualize fragments in context
- Affinity distribution analysis
- Fragment extraction pipeline

#### 💊 **Drug Design**
- Generate molecular scaffolds
- Property-based filtering
- Synthesizability assessment
- Export design templates

#### 📈 **Feature Evolution**
- Track feature importance over AL cycles
- Protocol phase visualization
- Statistical summaries
- Interactive exploration

#### 📉 **Distribution Analysis**
- Ridge plots of performance distributions
- Compare against random/MW baselines
- Dataset-specific visualizations

#### 🔬 **Molecular Analysis**
- Chemical space visualization (PCA/t-SNE/UMAP)
- Affinity prediction scatter plots
- Top compound identification
- Interactive filtering

#### 📄 **Publication Figures**
- Pre-formatted manuscript figures
- One-click generation
- High-resolution exports
- Customizable styling

#### 🧪 **Advanced Analytics**
- Cross-target heatmaps
- Feature stability analysis
- SAR analysis
- Chemical pattern recognition

---

## 🔧 Configuration Options

### Sidebar Settings

**Analysis Parameters:**
- `Top Features`: Number of top SHAP features to analyze (3-20)
- `Molecules per Feature`: Maximum molecules to extract per feature (5-50)

**Plot Configuration:**
- `Base Font Size`: 8-16 pt
- `Title Font Size`: 10-20 pt
- `Label Font Size`: 8-18 pt
- `DPI`: 100-500 (for exports)
- `Show Grid`: Enable/disable gridlines
- `Show Error Bars`: Toggle error bands

---

## 📖 Key Features

### Chemical Fragment Mapping

The fragment mapper extracts chemical substructures corresponding to important SHAP features:

```python
# Example workflow
mapper = ChemicalFragmentMapper(
    analysis_results=shap_data,
    dataset_df=molecule_df,
    fingerprint_func=smiles_to_ecfp8
)

fragments = mapper.extract_fragments_for_features(
    target='TYK2',
    protocol='ucb-exploit-heavy',
    top_n=10,
    max_mols=20
)
```

**Output includes:**
- Isolated fragment structures
- Parent molecule highlights
- Affinity distributions
- Fragment SMILES strings

### Protocol Performance Comparison

Compare different AL strategies with:
- Learning curves (compounds acquired vs. recall)
- Final performance distributions
- Statistical significance testing
- Baseline comparisons (random, MW-based)

### Publication Figure Generation

Generate manuscript-ready figures:
```python
# Similarity distributions
create_similarity_distribution_plot(df, datasets, filename='sim.png')

# Property boxplots
create_property_distribution_boxplots(props_df, datasets, filename='props.png')

# Performance summary (3-panel)
create_performance_summary_figure(results_df, datasets, filename='perf.png')

# Comprehensive heatmap
create_full_performance_heatmap(results_df, filename='heatmap.png')
```

---

## 🧪 Advanced Usage

### Custom Fingerprint Functions

Add your own molecular representations:

```python
def custom_fingerprint(smiles_df):
    """Custom fingerprint generator."""
    # Your implementation
    return np.array(fingerprints)

# Use in analysis
mapper = ChemicalFragmentMapper(
    analysis_results=data,
    dataset_df=df,
    fingerprint_func=custom_fingerprint
)
```

### Extending SAR Analysis

```python
analyzer = SARAnalyzer(dataset_df)

# R-group decomposition
core, r_groups = analyzer.analyze_r_groups()

# Activity cliff detection
cliffs = analyzer.analyze_activity_cliffs(activity_threshold=1.0)
```

### Custom Visualization Pipelines

```python
# Chemical space with custom parameters
fig, plot_df = visualize_chemical_space(
    smiles_df=df,
    color_by='affinity',
    method='UMAP',
    fp_type='ECFP',
    fp_radius=3,
    fp_bits=2048
)
```

---

## 📦 Data Sources

### Demo Dataset (Zenodo)

The demo data is automatically downloaded from:
```
https://zenodo.org/records/17935028
```

**Includes:**
- `TYK2_sorted.csv`: TYK2 kinase inhibitors
- `USP7_sorted.csv`: USP7 deubiquitinase inhibitors
- `D2R_composite.csv`: Dopamine D2 receptor ligands
- `MPRO_sorted.csv`: SARS-CoV-2 main protease inhibitors
- `recall_combined_results.csv`: AL performance metrics
- `mw_comparison_metrics.csv`: Molecular weight baseline
- Multiple `.pkl` files with SHAP analyses

---

## 🎨 Customization

### Color Schemes

Modify dataset/fingerprint colors in the code:

```python
DATASET_COLORS = {
    'TYK2': '#1f77b4',
    'USP7': '#ff7f0e',
    'D2R': '#2ca02c',
    'MPRO': '#d62728'
}

FP_COLORS = {
    'ECFP': '#5D69B1',
    'MACCS': '#52BCA3',
    'ChemBERTa': '#E506DA'
}
```

### Plot Styling

Apply consistent styles:

```python
FONT_SIZES = {
    'title': 20,
    'label': 18,
    'tick': 16,
    'legend': 16,
    'annotation': 14
}

def apply_plot_style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.5)
```

---

## 🐛 Troubleshooting

### Common Issues

**"No data loaded"**
- Click "Load Demo Dataset" in sidebar
- Or upload CSV/PKL files manually

**ChemBERTa fails**
- ChemBERTa requires GPU for reasonable performance
- Disable in similarity analysis if unavailable

**Fragment extraction shows "No SMILES Extracted"**
- Some features may not map to discrete fragments
- Check ECFP parameters (radius, bits)
- Ensure SMILES are valid

**UMAP not available**
- Install with: `pip install umap-learn`
- Falls back to PCA if unavailable

### Performance Optimization

For large datasets:
- Reduce sample size in similarity analysis
- Use ECFP/MACCS instead of ChemBERTa
- Filter molecules before visualization
- Enable caching with `@st.cache_data`

