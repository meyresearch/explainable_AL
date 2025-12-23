import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm


def smiles_to_ecfp8(smiles_df, smiles_column_name):
    """Convert SMILES in a DataFrame column to ECFP (Morgan) bit vectors.

    Parameters
    ----------
    smiles_df : pandas.DataFrame
        DataFrame containing a SMILES column.
    smiles_column_name : str
        Name of the column with SMILES strings.

    Returns
    -------
    numpy.ndarray
        Array of fingerprint vectors, shape (N, nBits).
    """
    fingerprints = []
    for smiles in tqdm(smiles_df[smiles_column_name], desc="Processing SMILES"):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=4, nBits=4096)
            fp_array = np.zeros((0,), dtype=np.int8)
            AllChem.DataStructs.ConvertToNumpyArray(fp, fp_array)
            fingerprints.append(fp_array)
    return np.array(fingerprints)
