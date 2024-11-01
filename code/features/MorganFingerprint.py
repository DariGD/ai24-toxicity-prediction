# Importing Libraries
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

def morgan_fingerprint(smile: str, radius: int=2, nBits: int=1024)-> list:
    """
    Функция позволяет переводить SMILES в Morgan Fingerprints
    """
    # Переводим Smiles в объект rdkit
    mol = Chem.MolFromSmiles(smile)

    if mol is None:
        raise ValueError("Неверный SMILES")

    # Создаем объект MorganFingerprint
    morganfp=AllChem.GetMorganFingerprintAsBitVect(mol,useChirality=True, radius=radius, nBits = nBits)
    
    # Превращаем объект MorganFingerprint в numpy массив
    mfpvector = np.array(morganfp)
    return mfpvector

