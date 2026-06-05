import os
import torch
import numpy as np
import pandas as pd
import random
import itertools
import networkx as nx
from torch_geometric.data import Data, InMemoryDataset
from rdkit import Chem
from Bio.PDB import PDBParser
from sklearn.metrics import (
    matthews_corrcoef, roc_auc_score, precision_score, recall_score,
    f1_score, accuracy_score, precision_recall_curve, auc, mean_squared_error,
    confusion_matrix, average_precision_score
)
import torch.nn.functional as F
from Bio.PDB import Vector, calc_dihedral

def set_seed(seed):
    # 1. 设置常规随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 2. 强制 CuDNN 确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 3. [关键修复] 禁用非确定性的 Attention 加速内核
    # Transformer 模块默认使用的 Flash/MemoryEfficient Attention 可能导致结果波动
    if hasattr(torch.backends, 'cuda'):
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            print("[Info] Disabled non-deterministic Flash/Mem-Efficient Attention.")
        except AttributeError:
            pass # 旧版本 PyTorch 可能没有这些属性

    # 4. 强制 PyTorch 确定性算法 (warn_only=False 确保如果有漏网之鱼会直接报错)
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        print("[Info] Deterministic algorithms enabled (Strict mode).")
    except AttributeError:
        pass
    
    # 5. 设置环境变量 (再次确保)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set {allowable_set}:")
    return [x == s for s in allowable_set]

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
                                           'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co',
                                           'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn',
                                           'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), list(range(11))) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), list(range(11))) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), list(range(11))) +
                    [atom.GetIsAromatic()])

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smile}")
    c_size = mol.GetNumAtoms()
    features = [atom_features(atom)/sum(atom_features(atom)) for atom in mol.GetAtoms()]
    features = np.array(features, dtype=np.float32)
    edges = [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol.GetBonds()]
    g = nx.Graph(edges).to_directed()
    edge_index = [[e1, e2] for e1,e2 in g.edges]
    return c_size, features, edge_index

def get_smiles_graph(dataset_path):
    df_Molecules = pd.read_excel(os.path.join(dataset_path, "Molecule.xlsx"))
    Molecules = set(df_Molecules['SMILES'].values)
    smile_graph = {}
    for smile in Molecules:
        try:
            smile_graph[smile] = smile_to_graph(smile)
        except:
            print(f"Error parsing SMILES: {smile}")
    return smile_graph

SMILES_CACHE_PATH = "data/processed/cache/smiles_graph.pt"

def get_smiles_graph_cached(dataset_path):
    if os.path.exists(SMILES_CACHE_PATH):
        return torch.load(SMILES_CACHE_PATH, weights_only=False)
    print("[Cache] First run, Building and caching SMILES graph...")
    smile_graph = get_smiles_graph(dataset_path)          
    os.makedirs(os.path.dirname(SMILES_CACHE_PATH), exist_ok=True)
    torch.save(smile_graph, SMILES_CACHE_PATH)
    return smile_graph

def rna2D_from_dot(seq, dot_bracket):
    L = min(len(seq), len(dot_bracket))
    seq = seq[:L]
    dot_bracket = dot_bracket[:L]
    
    base_dict = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}
    indices = torch.tensor([base_dict.get(b.upper(), 4) for b in seq], dtype=torch.long)
    x = F.one_hot(indices, num_classes=5).float()

    stack_round, stack_square = [], []
    edge_index, edge_type = [] , []
    edge_type_map = {'link': 0, ('C', 'G'): 1, ('A', 'U'): 2, ('G', 'U'): 3, 
                     ('A', 'G'): 4, ('U', 'U'): 5, ('C', 'C'): 6, ('A', 'A'): 7, 'unknown': 8}

    for i in range(L - 1):
        edge_index.extend([[i, i + 1], [i + 1, i]])  
        edge_type.extend([edge_type_map['link'], edge_type_map['link']])  

    for i in range(L):
        c = dot_bracket[i]
        if c == '(': 
            stack_round.append(i)
        elif c == ')':
            if stack_round:
                j = stack_round.pop()
                if i < L and j < L: 
                    sorted_bases = tuple(sorted([seq[i].upper(), seq[j].upper()]))
                    pair_type = edge_type_map.get(sorted_bases, edge_type_map['unknown'])
                    edge_index.extend([[i, j], [j, i]])
                    edge_type.extend([pair_type, pair_type])
        elif c == '[': 
            stack_square.append(i)
        elif c == ']':
            if stack_square:
                j = stack_square.pop()
                if i < L and j < L:
                    sorted_bases = tuple(sorted([seq[i].upper(), seq[j].upper()]))
                    pair_type = edge_type_map.get(sorted_bases, edge_type_map['unknown'])
                    edge_index.extend([[i, j], [j, i]])
                    edge_type.extend([pair_type, pair_type])

    if len(edge_index) > 0:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_type, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros(0, dtype=torch.long)
        
    return x, edge_index, edge_type

def calculate_torsion_angles(res, prev_res=None, next_res=None):
    angles = [0.0] * 6 
    try:
        if prev_res and 'O3\'' in prev_res and 'P' in res and 'O5\'' in res and 'C5\'' in res:
            v1, v2, v3, v4 = prev_res['O3\''].get_vector(), res['P'].get_vector(), res['O5\''].get_vector(), res['C5\''].get_vector()
            angles[0] = calc_dihedral(v1, v2, v3, v4)
        if all(a in res for a in ['P', 'O5\'', 'C5\'', 'C4\'']):
            v1, v2, v3, v4 = [res[a].get_vector() for a in ['P', 'O5\'', 'C5\'', 'C4\'']]
            angles[1] = calc_dihedral(v1, v2, v3, v4)
        if all(a in res for a in ['O5\'', 'C5\'', 'C4\'', 'C3\'']):
            v1, v2, v3, v4 = [res[a].get_vector() for a in ['O5\'', 'C5\'', 'C4\'', 'C3\'']]
            angles[2] = calc_dihedral(v1, v2, v3, v4)
        if all(a in res for a in ['C5\'', 'C4\'', 'C3\'', 'O3\'']):
            v1, v2, v3, v4 = [res[a].get_vector() for a in ['C5\'', 'C4\'', 'C3\'', 'O3\'']]
            angles[3] = calc_dihedral(v1, v2, v3, v4)
        if next_res and 'C4\'' in res and 'C3\'' in res and 'O3\'' in res and 'P' in next_res:
            v1, v2, v3, v4 = res['C4\''].get_vector(), res['C3\''].get_vector(), res['O3\''].get_vector(), next_res['P'].get_vector()
            angles[4] = calc_dihedral(v1, v2, v3, v4)
        if next_res and 'C3\'' in res and 'O3\'' in res and 'P' in next_res and 'O5\'' in next_res:
            v1, v2, v3, v4 = res['C3\''].get_vector(), res['O3\''].get_vector(), next_res['P'].get_vector(), next_res['O5\''].get_vector()
            angles[5] = calc_dihedral(v1, v2, v3, v4)
    except: pass
    return angles

def rna3D_from_pdb(rna_id, pdb_dir='PDB'):
    pdb_file = os.path.join(pdb_dir, f'{rna_id}.pdb')
    if not os.path.exists(pdb_file): return None, None

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(rna_id, pdb_file)
    
    atom_coords, atom_types = [], []
    atom_mapping = {'P': 0, 'O': 1, 'N': 2, 'C': 3, 'H': 4}
    res_coords, res_torsions = [], []

    for model in structure:
        for chain in model:
            residues = list(chain.get_residues())
            for i, res in enumerate(residues):
                for atom in res:
                    atom_coords.append(atom.get_coord())
                    atom_types.append(atom_mapping.get(atom.get_name().strip()[0], 5))
                
                if 'C4\'' in res:
                    res_coords.append(res['C4\''].get_coord())
                    prev_res = residues[i-1] if i > 0 else None
                    next_res = residues[i+1] if i < len(residues)-1 else None
                    res_torsions.append(calculate_torsion_angles(res, prev_res, next_res))

    if not atom_coords: return None, None

    atom_x = torch.tensor(np.concatenate([atom_coords, np.eye(6)[atom_types]], axis=1), dtype=torch.float)
    atom_dist = np.sqrt(((np.array(atom_coords)[:, None] - np.array(atom_coords)) ** 2).sum(-1))
    atom_ei = torch.tensor(np.argwhere((atom_dist < 5.0) & (atom_dist > 0)).T, dtype=torch.long)
    atom_ea = torch.tensor(atom_dist[atom_ei[0], atom_ei[1]], dtype=torch.float).unsqueeze(1)
    
    res_x = torch.tensor(np.concatenate([res_coords, res_torsions], axis=1), dtype=torch.float)
    res_dist = np.sqrt(((np.array(res_coords)[:, None] - np.array(res_coords)) ** 2).sum(-1))
    res_ei = torch.tensor(np.argwhere((res_dist < 10.0) & (res_dist > 0)).T, dtype=torch.long)
    res_ea = torch.tensor(res_dist[res_ei[0], res_ei[1]], dtype=torch.float).unsqueeze(1)

    return Data(x=atom_x, edge_index=atom_ei, edge_attr=atom_ea), \
           Data(x=res_x, edge_index=res_ei, edge_attr=res_ea)

def rna3D_from_pdb_cached(rna_id, pdb_dir, dataset_path):
    cache_dir = os.path.join("data/processed/cache", f"rna3d_multiscale_{os.path.basename(dataset_path)}")
    cache_path = os.path.join(cache_dir, f"{rna_id}.pt")
    os.makedirs(cache_dir, exist_ok=True)
    if os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=False)
    res = rna3D_from_pdb(rna_id, pdb_dir)
    torch.save(res, cache_path)
    return res

def get_kmer_dict(k_value):
    bases = ['A', 'U', 'C', 'G']
    all_k_mers = [''.join(p) for p in itertools.product(bases, repeat=k_value)]
    return {kmer: idx + 1 for idx, kmer in enumerate(all_k_mers)}

def get_kmer_index(sequence, kmer_dict, k_value, max_seq_len):
    k_ls = [kmer_dict.get(sequence[i:i + k_value], 0) for i in range(len(sequence) - k_value + 1)]
    k_ls += [0] * (max_seq_len - len(k_ls))
    return np.array(k_ls[:max_seq_len], dtype=np.int64)

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='data', dataset='default_dataset',
                 xd=None, xt=None, y=None, k_mer_features=None,
                 transform=None, pre_transform=None, smile_graph=None,
                 data_list=None):
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.data_list = data_list
        self.data, self.slices = self.load_or_process_data()

    def load_or_process_data(self):
        if os.path.exists(self.processed_paths[0]):
            return torch.load(self.processed_paths[0], weights_only=False)
        else:
            if self.data_list is None:
                raise ValueError("data_list must be provided to process new dataset in offline mode.")
            data, slices = self.collate(self.data_list)
            torch.save((data, slices), self.processed_paths[0])
            return data, slices

    @property
    def raw_file_names(self): return [f"{self.dataset}.csv"]
    @property
    def processed_file_names(self): return [f"{self.dataset}.pt"]
    def download(self): pass
    def _download(self): pass
    def _process(self):
        if not os.path.exists(self.processed_dir): os.makedirs(self.processed_dir)

def get_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    try:
        auc_score = float(roc_auc_score(y_true, y_pred))
    except: auc_score = 0.5
    aupr_score = float(average_precision_score(y_true, y_pred))
    
    pred_label = (y_pred >= 0.5).astype(int)
    f1 = float(f1_score(y_true, pred_label, zero_division=0))
    acc = float(accuracy_score(y_true, pred_label))
    rec = float(recall_score(y_true, pred_label, zero_division=0))
    prec = float(precision_score(y_true, pred_label, zero_division=0))
    
    tn = ((pred_label == 0) & (y_true == 0)).sum()
    fp = ((pred_label == 1) & (y_true == 0)).sum()
    spec = float(tn / (tn + fp + 1e-7))
    
    return [round(auc_score,4), round(aupr_score,4), round(f1,4), round(acc,4), round(rec,4), round(spec,4), round(prec,4)]
