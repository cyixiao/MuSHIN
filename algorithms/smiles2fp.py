import utils
import torch
import numpy as np
import deepchem as dc
import rdkit.Chem as Chem
from rdkit.Chem import AllChem

def smiles_to_fingerprint(smiles, radius=2, nBits=64):
    '''
    Convert SMILES strings to Morgan fingerprints.
    :param smiles: List of SMILES strings.
    :param radius: Radius used (default is 2).
    :param nBits: Number of bits used (default is 64).
    :return: Morgan fingerprints.
    '''
    features = []

    original_stderr, devnull = utils.suppress_cpp_warnings()
    try:
        for smile_item in smiles:
            mol = Chem.MolFromSmiles(smile_item) # Convert SMILES to a molecule
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, useFeatures=False) # Generate Morgan fingerprints
            fp_bits = fp.ToBitString() # Convert the fingerprints to a bit string
            finger_print = np.array(list(map(int, fp_bits))).astype(float).reshape(1, -1) # Convert the bit string to a numpy array
            features.append(finger_print) # Append the fingerprint to the features list
    finally:
        utils.restore_stderr(original_stderr, devnull)
    return torch.tensor(np.vstack(features), dtype=torch.float32)

def Tanimoto_smi(smiles_list):
    '''
    Calculate the Tanimoto similarity between SMILES strings.
    :param smiles_list: List of SMILES strings.
    :return: Tanimoto similarity matrix.
    '''
    def _compute(data_1, data_2):
        norm_1 = (data_1 ** 2).sum(axis=1).reshape(data_1.shape[0], 1)
        norm_2 = (data_2 ** 2).sum(axis=1).reshape(data_2.shape[0], 1)
        prod = data_1.dot(data_2.T)
        return prod / (norm_1 + norm_2.T - prod)

    fps = smiles_to_fingerprint(smiles_list)
    smi = np.ones((len(fps), len(fps)))
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            smi[i, j] = _compute(fps[i], fps[j])
            smi[j, i] = smi[i, j]
    return smi