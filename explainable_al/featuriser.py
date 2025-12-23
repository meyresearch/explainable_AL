"""Featuriser utilities: ECFP, MACCS, ChemBERTa wrappers.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from tqdm import tqdm
import torch


def smiles_to_ecfp8_df(smiles_df, smiles_column_name="SMILES", radius=4, nBits=4096):
    """Convert a DataFrame column of SMILES to ECFP fingerprints (numpy array).

    Returns an array of shape (N, nBits).
    """
    fingerprints = []
    for smiles in tqdm(smiles_df[smiles_column_name], desc="Processing SMILES"):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
            arr = np.zeros((nBits,), dtype=np.int8)
            AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
            fingerprints.append(arr)
        else:
            fingerprints.append(np.zeros((nBits,), dtype=np.int8))
    return np.vstack(fingerprints)


def get_maccs_from_smiles_list(smiles_list):
    """Return MACCS keys for a list of SMILES (list of arrays).
    """
    fps = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            fp = MACCSkeys.GenMACCSKeys(mol)
            arr = np.array(list(fp.ToBitString()), dtype=np.int8)
            fps.append(arr)
        else:
            fps.append(np.zeros(167, dtype=np.int8))
    return np.array(fps)


def load_chemberta_embeddings(npz_file):
    import numpy as _np

    data = _np.load(npz_file)
    if 'embeddings' in data:
        return data['embeddings']
    elif 'arr_0' in data:
        return data['arr_0']
    else:
        raise ValueError('No embeddings key found in npz file')


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
