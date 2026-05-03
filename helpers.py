import pandas as pd
from rdkit import Chem
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from rdkit import Chem, RDLogger
from rdkit.Chem import Lipinski, Descriptors, Crippen,rdMolDescriptors, GraphDescriptors, QED
from rdkit.Chem import Fragments, AllChem
from rdkit.Chem.rdMolDescriptors import (
    CalcTPSA, CalcKappa1, CalcKappa2, CalcKappa3,
    CalcChi0v, CalcChi1v, CalcChi2v, CalcChi3v, CalcChi4v,
    CalcChi0n, CalcChi1n, CalcChi2n, CalcChi3n, CalcChi4n,
)

from rdkit.Chem import AllChem
from rdkit.Chem import Crippen, Descriptors, Fragments, GraphDescriptors, QED, rdMolDescriptors
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F
import torch

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

def apply_featurizer(df, featurizer, col_name_smiles='SMILES', features_name=None, return_concat=False):
    """
    Apply a featurizer function to a SMILES column in a DataFrame and return the resulting feature matrix.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing a column with SMILES strings.
        
    featurizer : callable
        A function applied to each SMILES string. It must return a fixed-length
        iterable (e.g., list, tuple, numpy array) of numerical features.
        
    col_name_smiles : str, default='SMILES'
        Name of the column in `df` containing SMILES strings.
        
    features_name : str or list of str, optional, default=None
        If a string is provided, feature columns will be named
        `{features_name}_0`, `{features_name}_1`, ..., `{features_name}_n`.
        
        If a list of strings is provided, it must match the number of generated
        features exactly and will be used as column names.
        
        If None, columns will be named `feature_0`, `feature_1`, ..., `feature_n`.
        
    return_concat : bool, default=False
        If True, returns the original DataFrame concatenated with the generated
        feature columns. If False, returns only the feature DataFrame.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the generated features, or the concatenation
        of the original DataFrame and the features if `return_concat=True`.

    Raises
    ------
    ValueError
        If `features_name` is a list and its length does not match the number
        of generated features.

    Notes
    -----
    - The `featurizer` must return a fixed-length iterable for every input.
      Inconsistent feature lengths will raise errors when constructing the
      DataFrame.
    - The index of the returned DataFrame matches the index of the input `df`.
    - This function applies the featurizer row-wise using `pandas.Series.apply`,
      which may be slow for very large datasets.
    """

    features_df = pd.DataFrame(df[col_name_smiles].apply(featurizer).tolist(), index=df.index)
    
    if isinstance(features_name, str):
        features_df.columns = [f'{features_name}_{i}' for i in range(features_df.shape[1])]
    elif isinstance(features_name, list):
        if len(features_name) != features_df.shape[1]:
            raise ValueError(f"Length of features_name ({len(features_name)}) does not match number of features ({features_df.shape[1]})")
        features_df.columns = features_name
    else:
        features_df.columns = [f'feature_{i}' for i in range(features_df.shape[1])]
    
    if return_concat:
        return pd.concat([df, features_df], axis=1)
    else:
        return features_df
    
def smiles2physchem_extended(smiles: str) -> list:
    """
    Compute an extended set of physicochemical and topological descriptors from a SMILES string.

    Intentionally includes correlated descriptor groups to serve as a starting point
    for feature selection exercises.

    Parameters
    ----------
    smiles : str
        A SMILES string representing a chemical molecule.

    Returns
    -------
    list
        A list of descriptor values in the order defined by DESCRIPTOR_NAMES.
        Returns a list of None values if the SMILES cannot be parsed.
    """
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return [None] * len(DESCRIPTOR_NAMES)

    AllChem.ComputeGasteigerCharges(molecule)

    def count_atoms(symbol):
        return sum(1 for atom in molecule.GetAtoms() if atom.GetSymbol() == symbol)

    return [
        # Lipophilicity
        Crippen.MolLogP(molecule),
        Crippen.MolMR(molecule),
        Descriptors.MolLogP(molecule),
        # Molecular Weight & Size
        Descriptors.MolWt(molecule),
        Descriptors.HeavyAtomMolWt(molecule),
        Descriptors.HeavyAtomCount(molecule),
        molecule.GetNumAtoms(),
        rdMolDescriptors.CalcNumHeavyAtoms(molecule),
        # Hydrogen Bonding
        Descriptors.NumHDonors(molecule),
        Descriptors.NumHAcceptors(molecule),
        rdMolDescriptors.CalcNumHBD(molecule),
        rdMolDescriptors.CalcNumHBA(molecule),
        # Polarity & Surface Area
        CalcTPSA(molecule),
        rdMolDescriptors.CalcLabuteASA(molecule),
        Descriptors.PEOE_VSA1(molecule),
        Descriptors.PEOE_VSA2(molecule),
        Descriptors.PEOE_VSA3(molecule),
        Descriptors.PEOE_VSA4(molecule),
        Descriptors.PEOE_VSA5(molecule),
        Descriptors.PEOE_VSA6(molecule),
        Descriptors.SlogP_VSA1(molecule),
        Descriptors.SlogP_VSA2(molecule),
        Descriptors.SlogP_VSA3(molecule),
        Descriptors.SMR_VSA1(molecule),
        Descriptors.SMR_VSA2(molecule),
        Descriptors.SMR_VSA3(molecule),
        # Flexibility & Shape
        Descriptors.NumRotatableBonds(molecule),
        rdMolDescriptors.CalcNumRotatableBonds(molecule),
        CalcKappa1(molecule),
        CalcKappa2(molecule),
        CalcKappa3(molecule),
        # Connectivity / Topological Indices
        CalcChi0v(molecule),
        CalcChi1v(molecule),
        CalcChi2v(molecule),
        CalcChi3v(molecule),
        CalcChi4v(molecule),
        CalcChi0n(molecule),
        CalcChi1n(molecule),
        CalcChi2n(molecule),
        CalcChi3n(molecule),
        CalcChi4n(molecule),
        GraphDescriptors.Ipc(molecule),
        GraphDescriptors.BertzCT(molecule),
        GraphDescriptors.BalabanJ(molecule),
        rdMolDescriptors.CalcHallKierAlpha(molecule),
        # Ring Systems
        Descriptors.NumAromaticRings(molecule),
        Descriptors.NumAliphaticRings(molecule),
        Descriptors.NumSaturatedRings(molecule),
        rdMolDescriptors.CalcNumAromaticRings(molecule) + rdMolDescriptors.CalcNumAliphaticRings(molecule),
        Descriptors.NumAromaticCarbocycles(molecule),
        Descriptors.NumAromaticHeterocycles(molecule),
        Descriptors.NumAliphaticCarbocycles(molecule),
        Descriptors.NumAliphaticHeterocycles(molecule),
        Descriptors.NumSaturatedCarbocycles(molecule),
        Descriptors.NumSaturatedHeterocycles(molecule),
        rdMolDescriptors.CalcNumRings(molecule),
        # Atom-Type Counts
        Descriptors.NumHeteroatoms(molecule),
        count_atoms('N'),
        count_atoms('O'),
        count_atoms('S'),
        count_atoms('F') + count_atoms('Cl') + count_atoms('Br') + count_atoms('I'),
        count_atoms('F'),
        count_atoms('Cl'),
        count_atoms('Br'),
        rdMolDescriptors.CalcFractionCSP3(molecule),
        # Charge & Electronic
        Descriptors.MaxPartialCharge(molecule),
        Descriptors.MinPartialCharge(molecule),
        Descriptors.MaxAbsPartialCharge(molecule),
        Descriptors.MinAbsPartialCharge(molecule),
        # Drug-likeness
        QED.qed(molecule),
        rdMolDescriptors.CalcNumLipinskiHBA(molecule),
        rdMolDescriptors.CalcNumLipinskiHBD(molecule),
        len(Chem.FindMolChiralCenters(molecule, includeUnassigned=True)),
        rdMolDescriptors.CalcNumSpiroAtoms(molecule),
        rdMolDescriptors.CalcNumBridgeheadAtoms(molecule),
        # Fragments
        Fragments.fr_NH0(molecule),
        Fragments.fr_NH1(molecule),
        Fragments.fr_NH2(molecule),
        Fragments.fr_Al_OH(molecule),
        Fragments.fr_Al_OH_noTert(molecule),
        Fragments.fr_Ar_OH(molecule),
        Fragments.fr_ArN(molecule),
        Fragments.fr_Ar_NH(molecule),
        Fragments.fr_C_O(molecule),
        Fragments.fr_C_O_noCOO(molecule),
        Fragments.fr_COO(molecule),
        Fragments.fr_COO2(molecule),
        Fragments.fr_Al_COO(molecule),
        Fragments.fr_Ar_COO(molecule),
        Fragments.fr_halogen(molecule),
        Fragments.fr_alkyl_halide(molecule),
        Fragments.fr_sulfonamd(molecule),
        Fragments.fr_SH(molecule),
        Fragments.fr_nitro(molecule),
        Fragments.fr_nitro_arom(molecule),
        Fragments.fr_nitrile(molecule),
        Fragments.fr_amide(molecule),
        Fragments.fr_ester(molecule),
        Fragments.fr_ether(molecule),
        Fragments.fr_ketone(molecule),
        Fragments.fr_aldehyde(molecule),
    ]

def spearman_corr_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the pairwise Spearman correlation matrix for all columns of X.

    Parameters
    ----------
    X : pd.DataFrame of shape (n_samples, n_features)
        Feature matrix. All columns must be numeric.

    Returns
    -------
    corr : pd.DataFrame of shape (n_features, n_features)
        Symmetric matrix of Spearman correlation coefficients in [-1, 1].
    """
    corr, _ = spearmanr(X)
    return pd.DataFrame(corr, index=X.columns, columns=X.columns)

def permutation_importance_df(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute permutation feature importance and return a tidy DataFrame.

    Parameters
    ----------
    model : fitted sklearn estimator
    X : pd.DataFrame of shape (n_samples, n_features)
    y : pd.Series of shape (n_samples,)
    n_repeats : int
        Number of times each feature is permuted.
    random_state : int

    Returns
    -------
    df_imp : pd.DataFrame
        Columns: 'feature', 'importance_mean', 'importance_std'.
        Sorted by importance_mean descending.
    """
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring='r2',
    )
    df_imp = pd.DataFrame({
        'feature':          X.columns,
        'importance_mean':  result.importances_mean,
        'importance_std':   result.importances_std,
    }).sort_values('importance_mean', ascending=False).reset_index(drop=True)
    return df_imp

# previous function from homework
THRESHOLD_CORR_FEATURES = 0.7
def clustered_forward_selection(
    model: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    corr_matrix: pd.DataFrame,
    corr_threshold: float = THRESHOLD_CORR_FEATURES,
    max_features: int = 10,
    cv: int = 5,
    random_state: int = 42,
) -> list:
    """
    Clustered forward feature selection.

    Features are first grouped by correlation using a threshold on the
    absolute Spearman correlation matrix. At each round only one candidate
    per cluster is considered (the one not yet selected from that cluster).
    The candidate that maximises CV R² is added to the selected set and its
    cluster is locked — no further candidates are drawn from it until all
    other clusters have contributed a feature.

    Parameters
    ----------
    X_train : pd.DataFrame of shape (n_samples, n_features)
    y_train : pd.Series of shape (n_samples,)
    corr_matrix : pd.DataFrame
        Precomputed Spearman correlation matrix (from spearman_corr_matrix).
    corr_threshold : float
        Features with |correlation| >= corr_threshold are placed in the
        same cluster.
    max_features : int
        Maximum number of features to select.
    cv : int
        Number of cross-validation folds.
    random_state : int

    Returns
    -------
    selected : list of str
        Ordered list of selected feature names.
    scores : list of float
        CV R² score after adding each selected feature.
    cluster_labels : dict
        Mapping from feature name to cluster id.
    """
    features = list(X_train.columns)
    abs_corr = corr_matrix.abs()

    # Step 1: assign each feature to a cluster
    cluster_labels = {}
    cluster_id     = 0
    unassigned     = set(features)

    for feat in features:
        if feat not in unassigned:
            continue
        correlated = set(
            abs_corr.index[abs_corr[feat] >= corr_threshold].tolist()
        ) & unassigned
        for f in correlated:
            cluster_labels[f] = cluster_id
        unassigned -= correlated
        cluster_id += 1

    # Step 2: greedy forward selection with one candidate per cluster
    selected         = []
    scores           = []
    locked_clusters  = set()

    while len(selected) < max_features:
        available_clusters = [
            cid for cid in set(cluster_labels.values())
            if cid not in locked_clusters
        ]
        if not available_clusters:
            break

        # one candidate per available cluster
        candidates = []
        for cid in available_clusters:
            cluster_features = [
                f for f, c in cluster_labels.items()
                if c == cid and f not in selected
            ]
            if not cluster_features:
                continue

            best_cluster_score   = -np.inf
            best_cluster_feature = None
            for f in cluster_features:
                score    = cross_val_score(
                    model, X_train[[f]], y_train,
                    cv=cv, scoring='r2',
                ).mean()
                if score > best_cluster_score:
                    best_cluster_score   = score
                    best_cluster_feature = f
            if best_cluster_feature is not None:
                candidates.append(best_cluster_feature)

        if not candidates:
            break

        # evaluate each candidate in context and pick the best one
        best_score   = -np.inf
        best_feature = None

        for feature in candidates:
            candidate_set = selected + [feature]
            cv_score      = cross_val_score(
                model, X_train[candidate_set], y_train,
                cv=cv, scoring='r2',
            ).mean()
            if cv_score > best_score:
                best_score   = cv_score
                best_feature = feature

        if best_feature is None:
            break

        selected.append(best_feature)
        scores.append(best_score)
        locked_clusters.add(cluster_labels[best_feature])

    return selected, scores, cluster_labels

def forward_selection(
    model: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_features: int = 10,
    cv: int = 5,
    random_state: int = 42,
) -> list:
    """
    Greedy forward feature selection using cross-validated R² as criterion.

    At each round the feature that maximises CV R² when added to the
    current selected set is chosen. Selection stops when max_features
    is reached.

    Parameters
    ----------
    X_train : pd.DataFrame of shape (n_samples, n_features)
    y_train : pd.Series of shape (n_samples,)
    max_features : int
        Maximum number of features to select.
    cv : int
        Number of cross-validation folds.
    random_state : int

    Returns
    -------
    selected : list of str
        Ordered list of selected feature names.
    scores : list of float
        CV R² score after adding each selected feature.
    """
    remaining = list(X_train.columns)
    selected  = []
    scores    = []

    while len(selected) < max_features and remaining:
        best_score   = -np.inf
        best_feature = None

        for feature in remaining:
            candidate = selected + [feature]
            cv_scores = cross_val_score(
                model, X_train[candidate], y_train,
                cv=cv, scoring='r2',
            )
            score = cv_scores.mean()
            if score > best_score:
                best_score   = score
                best_feature = feature

        selected.append(best_feature)
        remaining.remove(best_feature)
        scores.append(best_score)

    return selected, scores

def smiles2graph(smiles):
    """
    Convert a SMILES string to a dense graph representation.

    Returns
    -------
    X : np.ndarray, shape (N, 6)
        Node feature matrix. Row i contains the feature vector of atom i.
    A : np.ndarray, shape (N, N)
        Binary symmetric adjacency matrix (without self-loops).
        A[i, j] = 1 if atoms i and j share a bond.
        A[i, i] = 0 no self-loops.
    """
    # Define a mapping from bond types to integers
    order_string = {
        Chem.rdchem.BondType.SINGLE: 1,
        Chem.rdchem.BondType.DOUBLE: 2,
        Chem.rdchem.BondType.TRIPLE: 3,
        Chem.rdchem.BondType.AROMATIC: 4,
    }
    # Convert the SMILES string to an RDKit molecule object
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'RDKit could not parse SMILES: {smiles}')
    # Add hydrogen atoms to the molecule
    # mol = Chem.AddHs(mol)

    # Count the number of atoms in the molecule
    N = mol.GetNumAtoms()

    # --- Node feature matrix ---
    X = np.zeros((N, 54), dtype=np.float32)
    for atom in mol.GetAtoms():
        X[atom.GetIdx(),atom.GetAtomicNum()-1] = 1

    # --- Adjacency matrix ---
    A = np.zeros((N, N, 5), dtype=np.float32)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        order = bond.GetBondType()
        # Convert the bond type to an integer, if recognized
        if order in order_string:
            order = order_string[order]
        else:
            # Warn if an unrecognized bond type is encountered
            raise Warning("Ignoring bond order" + str(order))
        # Mark the presence of bonds in the adjacency matrix
        A[i, j, order] = 1
        A[j, i, order] = 1

    # Collapse the bond type dimension the adjacency matrix (do not add self-loops)
    A = np.sum(A, axis=-1)
    # Return the node feature matrix (X), the adjacency matrix (A)
    return X, A

class GCNLayer(nn.Module):
    """
    A single Graph Convolutional layer (Kipf & Welling, ICLR 2017).

    Update rule:
        H_out = activation( D_tilde^{-1/2} A_tilde D_tilde^{-1/2}  H  W )

    where A_tilde = A + I  (self-loops added) and D_tilde is its degree matrix.

    Parameters
    ----------
    in_features  : int  — input feature dimension
    out_features : int  — output feature dimension
    activation   : callable or None — default ReLU; pass None for a linear layer
    """

    def __init__(self, in_features, out_features, activation=F.relu):
        super().__init__()
        self.W = nn.Parameter(torch.empty(in_features, out_features))
        self.activation = activation
        nn.init.xavier_uniform_(self.W)  # Glorot initialisation
        
    def _normalised_adj(self, A):
        """Compute D_tilde^{-1/2} (A + I) D_tilde^{-1/2}."""
        N = A.size(0)
        A_tilde = A + torch.eye(N, device=A.device, dtype=A.dtype)   # self-loops
        d_tilde = A_tilde.sum(dim=1)                                  # row sums
        d_inv_sqrt = d_tilde.pow(-0.5)                                # d^{-1/2}
        # Broadcasting: (N,1) * (N,N) * (1,N)
        return d_inv_sqrt.unsqueeze(1) * A_tilde * d_inv_sqrt.unsqueeze(0)

    def forward(self, X, A):
        """
        Parameters
        ----------
        X : Tensor (N, in_features)   node feature matrix
        A : Tensor (N, N)             adjacency matrix (no self-loops)

        Returns
        -------
        Tensor (N, out_features)
        """
        A_norm = self._normalised_adj(A)    # (N, N)
        
        H_new = torch.einsum("ij,jk,kp->ip",
                             A_norm, X, self.W
                             )
        
        if self.activation is not None:
            H_new = self.activation(H_new)
        return H_new

class GCNModel(nn.Module):
    """
    Multi-layer GCN for graph-level regression.

    Architecture
    ------------
    GCNLayer(in → hidden) → ReLU
    [ GCNLayer(hidden → hidden) → ReLU ] × (n_layers - 1)
    Global pool  →  molecule embedding
    MLP(hidden → hidden/2 → out)
    """

    def __init__(self, in_features, hidden, out_features=1, n_layers=2, pool_fn=torch.mean):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNLayer(in_features, hidden))
        for _ in range(n_layers - 1):
            self.convs.append(GCNLayer(hidden, hidden))

        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, out_features),
        )
        self.pool_fn = pool_fn


    def forward(self, X, A):
        """
        Parameters
        ----------
        X : Tensor (N, in_features)  — node features
        A : Tensor (N, N)            — adjacency matrix (no self-loops)

        Returns
        -------
        Tensor (out_features,)  — graph-level prediction
        """
        h = X
        for conv in self.convs:
            h = conv(h, A)          # (N, hidden)

        # Global pooling: (N, hidden) → (hidden,)
        h_graph = self.pool_fn(h, dim=0)
        return self.readout(h_graph)  # (out_features,)

class GNNDataset(Dataset):
    """PyTorch Dataset for GNN data."""

    def __init__(self, smiles_list, labels):
        self.data = []
        n_skip = 0
        for smi, y in zip(smiles_list, labels):
            try:
                X, A = smiles2graph(smi)
                self.data.append((
                    torch.tensor(X),
                    torch.tensor(A),
                    torch.tensor(float(y)),
                ))
            except Exception:
                n_skip += 1
                print(smi)
        if n_skip:
            print(f'  Skipped {n_skip} unparseable SMILES.')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

DESCRIPTOR_NAMES = [
    # --- Lipophilicity (correlated group) ---
    'logp',                       # Wildman-Crippen octanol-water partition coefficient
    'molar_refractivity',         # Crippen molar refractivity
    'logp_rdkit',                 # RDKit built-in logP; slightly different implementation

    # --- Molecular Weight & Size (correlated group) ---
    'mol_wt',                     # Average molecular weight based on atomic masses
    'heavy_atom_mol_wt',          # Molecular weight of heavy (non-H) atoms only
    'heavy_atom_count',           # Number of non-hydrogen atoms
    'num_atoms',                  # Total atom count including explicit hydrogens
    'num_heavy_atoms',            # Heavy atom count via rdMolDescriptors

    # --- Hydrogen Bonding (correlated group) ---
    'hbond_donor',                # Lipinski H-bond donor count (-OH, -NH groups)
    'hbond_acceptor',             # Lipinski H-bond acceptor count (O, N atoms)
    'hbond_donor_rdkit',          # RDKit H-bond donor count; slightly stricter definition
    'hbond_acceptor_rdkit',       # RDKit H-bond acceptor count; slightly stricter definition

    # --- Polarity & Surface Area (correlated group) ---
    'tpsa',                       # Topological polar surface area
    'labute_asa',                 # Labute approximate surface area
    'peoe_vsa1',                  # Surface area contribution from atoms with PEOE partial charge in bin 1
    'peoe_vsa2',                  # Surface area contribution from atoms with PEOE partial charge in bin 2
    'peoe_vsa3',                  # Surface area contribution from atoms with PEOE partial charge in bin 3
    'peoe_vsa4',                  # Surface area contribution from atoms with PEOE partial charge in bin 4
    'peoe_vsa5',                  # Surface area contribution from atoms with PEOE partial charge in bin 5
    'peoe_vsa6',                  # Surface area contribution from atoms with PEOE partial charge in bin 6
    'slogp_vsa1',                 # Surface area contribution from atoms with SlogP contribution in bin 1
    'slogp_vsa2',                 # Surface area contribution from atoms with SlogP contribution in bin 2
    'slogp_vsa3',                 # Surface area contribution from atoms with SlogP contribution in bin 3
    'smr_vsa1',                   # Surface area contribution from atoms with SMR contribution in bin 1
    'smr_vsa2',                   # Surface area contribution from atoms with SMR contribution in bin 2
    'smr_vsa3',                   # Surface area contribution from atoms with SMR contribution in bin 3

    # --- Flexibility & Shape ---
    'num_rotatable_bonds',        # Number of rotatable bonds (Lipinski definition); measure of molecular flexibility
    'num_rotatable_bonds_rdkit',  # Number of rotatable bonds (RDKit definition)
    'kappa1',                     # Hall-Kier kappa index 1; encodes overall molecular shape and size
    'kappa2',                     # Hall-Kier kappa index 2; encodes degree of branching
    'kappa3',                     # Hall-Kier kappa index 3; encodes 3-atom branching patterns

    # --- Connectivity / Topological Indices (highly intercorrelated group) ---
    'chi0v',                      # Valence-weighted zeroth-order chi connectivity index
    'chi1v',                      # Valence-weighted first-order chi connectivity index
    'chi2v',                      # Valence-weighted second-order chi connectivity index
    'chi3v',                      # Valence-weighted third-order chi connectivity index
    'chi4v',                      # Valence-weighted fourth-order chi connectivity index
    'chi0n',                      # Non-H atom zeroth-order chi connectivity index
    'chi1n',                      # Non-H atom first-order chi connectivity index
    'chi2n',                      # Non-H atom second-order chi connectivity index
    'chi3n',                      # Non-H atom third-order chi connectivity index
    'chi4n',                      # Non-H atom fourth-order chi connectivity index
    'ipc',                        # Bonchev-Trinajstic information content index
    'bertz_ct',                   # Bertz complexity index
    'balaban_j',                  # Balaban J topological index; measures molecular branching and cyclicity
    'hallkier_alpha',             # Hall-Kier alpha correction; accounts for atom size deviations from carbon

    # --- Ring Systems (correlated group) ---
    'num_aromatic_rings',         # Number of aromatic rings
    'num_aliphatic_rings',        # Number of aliphatic (non-aromatic) rings
    'num_saturated_rings',        # Number of fully saturated rings; subset of num_aliphatic_rings
    'num_rings',                  # Total ring count (aromatic + aliphatic)
    'num_aromatic_carbocycles',   # Number of aromatic rings containing only carbon; subset of num_aromatic_rings
    'num_aromatic_heterocycles',  # Number of aromatic rings containing heteroatoms
    'num_aliphatic_carbocycles',  # Number of aliphatic carbocyclic rings; subset of num_aliphatic_rings
    'num_aliphatic_heterocycles', # Number of aliphatic rings containing heteroatoms
    'num_saturated_carbocycles',  # Number of saturated carbocyclic rings; subset of num_aliphatic_carbocycles
    'num_saturated_heterocycles', # Number of saturated heterocyclic rings; subset of num_aliphatic_heterocycles
    'ring_count',                 # Total ring count via rdMolDescriptors

    # --- Atom-Type Counts (correlated group) ---
    'num_heteroatoms',            # Total number of non-C, non-H atoms; sum of N, O, S, halogens etc.
    'num_nitrogen',               # Number of nitrogen atoms; subset of num_heteroatoms
    'num_oxygen',                 # Number of oxygen atoms; subset of num_heteroatoms
    'num_sulfur',                 # Number of sulfur atoms; subset of num_heteroatoms
    'num_halogens',               # Total number of halogen atoms (F+Cl+Br+I); subset of num_heteroatoms
    'num_fluorine',               # Number of fluorine atoms; subset of num_halogens
    'num_chlorine',               # Number of chlorine atoms; subset of num_halogens
    'num_bromine',                # Number of bromine atoms; subset of num_halogens
    'fraction_csp3',              # Fraction of sp3 carbons (Fsp3); measure of three-dimensionality

    # --- Charge & Electronic Properties ---
    'max_partial_charge',         # Maximum Gasteiger partial charge over all atoms
    'min_partial_charge',         # Minimum Gasteiger partial charge over all atoms
    'max_abs_partial_charge',     # Maximum absolute Gasteiger partial charge
    'min_abs_partial_charge',     # Minimum absolute Gasteiger partial charge

    # --- Drug-likeness ---
    'qed',                        # Quantitative estimate of drug-likeness (0-1); composite of multiple properties
    'num_lipinskihba',            # Lipinski H-bond acceptor count (explicit RDKit Lipinski definition)
    'num_lipinskihbd',            # Lipinski H-bond donor count (explicit RDKit Lipinski definition)
    'num_stereocenters',          # Number of stereocenters (assigned and unassigned)
    'num_spiro_atoms',            # Number of spiro atoms (atoms shared between two rings via one atom)
    'num_bridgehead_atoms',       # Number of bridgehead atoms (atoms shared between two rings via two+ bonds)

    # --- Fragment / Functional Group Counts (correlated with atom-type counts above) ---
    'fr_NH0',                     # Number of tertiary amines; subset of num_nitrogen
    'fr_NH1',                     # Number of secondary amines; subset of num_nitrogen
    'fr_NH2',                     # Number of primary amines; subset of num_nitrogen
    'fr_Al_OH',                   # Number of aliphatic hydroxyl groups; subset of num_oxygen
    'fr_Al_OH_noTert',            # Number of aliphatic hydroxyl groups excluding tertiary OH
    'fr_Ar_OH',                   # Number of aromatic hydroxyl groups (phenols); subset of num_oxygen
    'fr_ArN',                     # Number of N functional groups attached to aromatics; subset of num_nitrogen
    'fr_Ar_NH',                   # Number of aromatic amine groups
    'fr_C_O',                     # Number of C=O groups (aldehydes, ketones, acids, esters)
    'fr_C_O_noCOO',               # Number of C=O groups excluding carboxylic acids
    'fr_COO',                     # Number of carboxylate groups; subset of fr_C_O
    'fr_COO2',                    # Number of carboxylic acid groups
    'fr_Al_COO',                  # Number of aliphatic carboxylic acids; subset of fr_COO2
    'fr_Ar_COO',                  # Number of aromatic carboxylic acids; subset of fr_COO2
    'fr_halogen',                 # Number of halogen atoms via fragment matching
    'fr_alkyl_halide',            # Number of alkyl halide groups; subset of fr_halogen
    'fr_sulfonamd',               # Number of sulfonamide groups; subset of num_sulfur
    'fr_SH',                      # Number of thiol groups; subset of num_sulfur
    'fr_nitro',                   # Number of nitro groups
    'fr_nitro_arom',              # Number of aromatic nitro groups; subset of fr_nitro
    'fr_nitrile',                 # Number of nitrile groups; subset of num_nitrogen
    'fr_amide',                   # Number of amide groups
    'fr_ester',                   # Number of ester groups
    'fr_ether',                   # Number of ether groups
    'fr_ketone',                  # Number of ketone groups; subset of fr_C_O_noCOO
    'fr_aldehyde',                # Number of aldehyde groups; subset of fr_C_O_noCOO
]