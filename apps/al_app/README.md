# Active Learning Application

A Streamlit-based interactive application for active learning in drug discovery, enabling comparison of different acquisition strategies for molecular property prediction.

---

## 📋 Overview

This application provides a user-friendly interface for running active learning experiments on molecular datasets. It supports multiple acquisition strategies (UCB, PI, EI), different molecular representations (ECFP, MACCS, ChemBERTa), and various Gaussian Process kernels.

**Key Features:**
- **Interactive protocol configuration**: Choose pre-defined protocols or create custom acquisition strategies
- **Multiple molecular representations**: ECFP fingerprints, MACCS keys, ChemBERTa embeddings
- **Flexible GP kernels**: Tanimoto, RBF, Matern, Linear, Rational Quadratic
- **Real-time monitoring**: Progress tracking and live results updates
- **Comparative analysis**: Run multiple protocols side-by-side
- **Visualization**: Recall curves and performance metrics

---

### Running the Application

```bash
# Launch the Streamlit app
streamlit run apps/al_app/app.py

# Or from the project root
python -m streamlit run apps/al_app/app.py
```

---

## 📁 Application Structure

```
apps/al_app/
├── app.py                      # Main Streamlit application
├── active_learning_core.py     # Compatibility wrapper for AL core functions
├── plotting.py                 # Plotting utilities wrapper
├── utils.py                    # Utility functions wrapper
└── README.md                   # This file
```

**Note**: This application is a thin wrapper around the canonical implementations in `explainable_al/`. All core functionality is imported from the package.

---

## 📊 Usage Guide

### 1. Upload Dataset

**Required CSV format:**
- One column with SMILES strings (molecular structures)
- One column with target values (e.g., binding affinity, activity)

**Example:**
```csv
SMILES,affinity
CCO,5.2
c1ccccc1,4.8
CC(C)O,6.1
```

**Supported formats:**
- Standard CSV with header row
- SMILES strings in any column (selectable in UI)
- Continuous target values (regression task)

### 2. Define Task

- **Feature Column (X)**: Select the column containing SMILES strings
- **Target Column (Y)**: Select the column with target values

The application will automatically create:
- `top_2p`: Binary labels for top 2% compounds
- `top_5p`: Binary labels for top 5% compounds

### 3. Molecular Representation

Choose how molecules are represented:

#### **ECFP Fingerprints** (Recommended)
- Extended Connectivity Fingerprints
- Radius: 2, Bits: 2048 (default)
- Fast generation, works with Tanimoto kernel
- Good for most drug-like molecules

#### **MACCS Keys**
- 166-bit structural key descriptors
- Fixed-size, interpretable features
- Captures common substructures

#### **ChemBERTa Embeddings**
- Deep learning-based molecular embeddings
- Requires pre-computed `.npz` file
- High-dimensional (768D) representations
- Best for transfer learning scenarios

**Generating representations:**
1. Select representation type
2. Upload embeddings file (if using ChemBERTa)
3. Click "Generate Representations"
4. Wait for confirmation message

### 4. Surrogate Model

Choose a Gaussian Process kernel:

| Kernel | Description | Best For |
|--------|-------------|----------|
| **Tanimoto** | Molecular similarity kernel | ECFP/MACCS fingerprints |
| **RBF** | Radial basis function | Continuous embeddings |
| **Matern** | Smooth interpolation | General-purpose |
| **Linear** | Linear relationships | Simple baselines |
| **Rational Quadratic** | Multi-scale patterns | Complex landscapes |

**Recommendation**: Use Tanimoto kernel with ECFP fingerprints for molecular data.

### 5. Acquisition Protocol

#### **Pre-defined Protocols**

Two standard protocols are available:

**ucb-explore-heavy:**
```python
[("random", 60)] + [("explore", 30)] * 7 + [("exploit", 30)] * 3
```
- 1 cycle random initialization (60 compounds)
- 7 cycles exploration (uncertainty-focused)
- 3 cycles exploitation (prediction-focused)

**ucb-exploit-heavy:**
```python
[("random", 60)] + [("explore", 30)] * 3 + [("exploit", 30)] * 7
```
- 1 cycle random initialization (60 compounds)
- 3 cycles exploration
- 7 cycles exploitation (more greedy)

#### **Custom Protocol**

Enable the "Custom Protocol" checkbox to configure:

1. **Initial Set Size**: Number of random compounds to start (default: 60)
2. **Number of Cycles**: Total AL iterations (default: 10)
3. **Batch Size**: Compounds per cycle (default: 30)
4. **Acquisition Method**: UCB, PI, or EI

**UCB (Upper Confidence Bound):**
- Balance exploration vs. exploitation
- Configure number of exploration cycles
- Remaining cycles use exploitation

**PI (Probability of Improvement):**
- Maximize probability of finding better compounds
- Adjust `xi` parameter (default: 0.01)
- Higher `xi` → more exploration

**EI (Expected Improvement):**
- Maximize expected improvement over current best
- Adjust `xi` parameter (default: 0.01)
- Balances magnitude and probability of improvement

### 6. Run Experiments

1. Select one or more pre-defined protocols
2. Optionally enable and configure custom protocol
3. Click "Start Active Learning"
4. Monitor progress in real-time
5. View results table and recall curves

---

## 📈 Results and Metrics

### Performance Metrics

For each cycle, the application tracks:

| Metric | Description |
|--------|-------------|
| **Cycle** | AL iteration number (0 = random init) |
| **Method** | Acquisition strategy used (random/explore/exploit/PI/EI) |
| **Compounds acquired** | Cumulative molecules selected |
| **R²** | Coefficient of determination (model fit) |
| **Spearman** | Rank correlation (prediction quality) |
| **Recall (2%)** | Fraction of top 2% compounds found |
| **Recall (5%)** | Fraction of top 5% compounds found |

### Visualization

**Recall Curves:**
- X-axis: Compounds acquired
- Y-axis: Recall percentage
- Multiple protocols overlayed for comparison
- Shows learning efficiency over time

**Key Insights:**
- Steeper curves = more efficient protocols
- Compare final recall values
- Assess exploration-exploitation balance

---

## 🔧 Advanced Configuration

### Custom Protocol Design

Create complex acquisition strategies by mixing methods:

```python
custom_protocol = [
    ("random", 60),        # Initial random sample
    ("explore", 30),       # Exploration phase
    ("explore", 30),
    ("PI", 30, 0.05),      # Switch to PI with higher xi
    ("EI", 30, 0.01),      # Try EI
    ("exploit", 30),       # Final exploitation
]
```

### Kernel Configuration

For advanced users, kernels can be customized in `app.py`:

```python
# Example: Tanimoto with custom variance prior
from explainable_al.active_learning_core import TanimotoKernel
kernel = TanimotoKernel()

# Example: RBF with length scale prior
kernel = gpytorch.kernels.ScaleKernel(
    gpytorch.kernels.RBFKernel(
        lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0)
    )
)
```

---

## 📊 Example Workflow

### Complete Example

```python
# 1. Prepare dataset
import pandas as pd

data = pd.DataFrame({
    'SMILES': ['CCO', 'c1ccccc1', 'CC(C)O', ...],
    'affinity': [5.2, 4.8, 6.1, ...]
})
data.to_csv('molecules.csv', index=False)

# 2. Launch app
# streamlit run apps/al_app/app.py

# 3. In the UI:
# - Upload molecules.csv
# - Select SMILES → 'SMILES'
# - Select Target → 'affinity'
# - Choose ECFP Fingerprints
# - Generate Representations
# - Select Tanimoto Kernel
# - Choose protocols to compare
# - Start Active Learning

# 4. Results will show:
# - Live progress updates
# - Final results table
# - Recall curves comparing protocols
```

---



## 🔄 Integration with Package

This app uses the canonical implementations from `explainable_al/`:

```python
# Core AL functions
from explainable_al.active_learning_core import (
    TanimotoKernel,
    GPRegressionModel,
    train_gp_model,
    ucb_selection,
    pi_selection,
    ei_selection,
    run_active_learning_experiment,
)

# Utilities
from explainable_al.utils import (
    get_ecfp_fingerprints,
    get_maccs_keys,
    get_chemberta_embeddings,
    calculate_metrics,
)

# Plotting
from explainable_al.metrics_plots import make_plot_recall
```


