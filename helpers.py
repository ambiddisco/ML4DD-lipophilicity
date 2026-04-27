import pandas as pd
from rdkit import Chem
import numpy as np
import matplotlib.pyplot as plt
from copy import copy

from rdkit.Chem import AllChem
from rdkit.Chem import Crippen, Descriptors, Fragments, GraphDescriptors, QED, rdMolDescriptors

def compute_descriptors(mol):
# computes descriptors for a single RDKit molecule [WE CAN ADD MORE LATER IF WE WANT]
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        pass

    return {
        # Lipophilicity (correlated group)
        'logP_Crippen': Crippen.MolLogP(mol), # 1
        'logP_RDKit': Descriptors.MolLogP(mol), # 2
        'MolMR': Crippen.MolMR(mol), # 3

        # Molecular size (correlated group)
        'MolWt': Descriptors.MolWt(mol), # 4
        'HeavyAtomMolWt': Descriptors.HeavyAtomMolWt(mol), # 5
        'HeavyAtomCount': Descriptors.HeavyAtomCount(mol), # 6
        'NumAtoms': mol.GetNumAtoms(), # 7

        # Hydrogen bonding (correlated group)
        'HBDonors': Descriptors.NumHDonors(mol), # 8
        'HBAcceptors': Descriptors.NumHAcceptors(mol), # 9
        'HBDonors_RDKit': rdMolDescriptors.CalcNumHBD(mol), # 10
        'HBAcceptors_RDKit': rdMolDescriptors.CalcNumHBA(mol), # 11

        # Polarity and surface area (correlated group)
        'TPSA': rdMolDescriptors.CalcTPSA(mol), # 12
        'LabuteASA': rdMolDescriptors.CalcLabuteASA(mol), # 13

        # Flexibility and shape
        'NumRotBonds': Descriptors.NumRotatableBonds(mol), # 14
        'Kappa1': rdMolDescriptors.CalcKappa1(mol), # 15
        'Kappa2': rdMolDescriptors.CalcKappa2(mol), # 16
        'Kappa3': rdMolDescriptors.CalcKappa3(mol), # 17

        # Topological / connectivity indices (correlated group)
        'Chi0v': rdMolDescriptors.CalcChi0v(mol), # 18
        'Chi1v': rdMolDescriptors.CalcChi1v(mol), # 19
        'Chi2v': rdMolDescriptors.CalcChi2v(mol), # 20
        'Chi3v': rdMolDescriptors.CalcChi3v(mol), # 21
        'Chi0n': rdMolDescriptors.CalcChi0n(mol), # 22
        'Chi1n': rdMolDescriptors.CalcChi1n(mol), # 23
        'Chi2n': rdMolDescriptors.CalcChi2n(mol), # 24
        'Chi3n': rdMolDescriptors.CalcChi3n(mol), # 25
        'BertzCT': GraphDescriptors.BertzCT(mol), # 26
        'BalabanJ': GraphDescriptors.BalabanJ(mol), # 27
        'HallKierAlpha': GraphDescriptors.HallKierAlpha(mol), # 28

        # Ring systems (correlated group)
        'NumAromaticRings': Descriptors.NumAromaticRings(mol), # 29
        'NumAliphaticRings': Descriptors.NumAliphaticRings(mol), # 30
        'RingCount': rdMolDescriptors.CalcNumRings(mol), # 31

        # Atom-type counts
        'NumHeteroatoms': Descriptors.NumHeteroatoms(mol), # 32
        'NumNitrogen': sum(atom.GetAtomicNum() == 7 for atom in mol.GetAtoms()), # 33
        'NumOxygen': sum(atom.GetAtomicNum() == 8 for atom in mol.GetAtoms()), # 34
        'FractionCSP3': rdMolDescriptors.CalcFractionCSP3(mol), # 35

        # Electronic
        'MaxPartialCharge': Descriptors.MaxPartialCharge(mol), # 36
        'MinPartialCharge': Descriptors.MinPartialCharge(mol), # 37

        # Drug-likeness
        'QED': QED.qed(mol), # 38

        # Functional groups
        'fr_NH0': Fragments.fr_NH0(mol), # 39
        'fr_ArOH': Fragments.fr_Ar_OH(mol), # 40
    }

