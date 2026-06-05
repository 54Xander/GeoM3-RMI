import os
import copy
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.makedirs("data/processed/cache", exist_ok=True)

import torch
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel

from utils import (
    get_smiles_graph_cached,
    get_smiles_graph,
    get_kmer_index,
    get_kmer_dict,
    TestbedDataset,
    rna2D_from_dot,
    rna3D_from_pdb_cached
)

def get_bert_embeddings_batch(text_list, model_name, device, batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    all_embeddings = []
    print(f"Extracting BERT embeddings (Batch size: {batch_size})...")
    with torch.no_grad():
        for i in tqdm(range(0, len(text_list), batch_size)):
            batch_texts = [str(t) for t in text_list[i:i+batch_size]]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                               max_length=128, return_tensors="pt").to(device)
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu()
            all_embeddings.append(embeddings)
    return torch.cat(all_embeddings, dim=0)

def read_raw_data(dataset_path, n_splits=5, seed=42, val_size=0.1, split_mode='random'):
    """
    读取原始数据并进行划分。
    [CRITICAL FIX] 修复验证集泄露，并支持 Random, Cold RNA, Cold Drug, Cold Both 四种模式。
    """
    dataset_name = os.path.basename(dataset_path)
    df_Molecules = pd.read_excel(os.path.join(dataset_path, "Molecule.xlsx"))
    df_rnas = pd.read_excel(os.path.join(dataset_path, "RNA.xlsx")).set_index('RNA_ID')
    df_labels = pd.read_excel(os.path.join(dataset_path, "RNA-Molecule.xlsx"))

    # 统一列名检查
    label_col = 'label' if 'label' in df_labels.columns else 'Label'
    mol_id_col = 'Small molecule_ID' if 'Small molecule_ID' in df_labels.columns else 'Molecule_ID'

    all_folds_data = []

    # 1. 准备划分索引生成器
    splits = []
    
    if split_mode == 'random':
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(kf.split(df_labels, df_labels[label_col]))
        unique_rnas, unique_mols = None, None
    
    elif split_mode == 'cold_rna':
        unique_rnas = df_labels['RNA_ID'].unique()
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(kf.split(unique_rnas))
        unique_mols = None
    
    elif split_mode == 'cold_drug':
        unique_mols = df_labels[mol_id_col].unique()
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(kf.split(unique_mols))
        unique_rnas = None

    elif split_mode == 'cold_both':
        unique_rnas = df_labels['RNA_ID'].unique()
        unique_mols = df_labels[mol_id_col].unique()
        kf_rna = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        kf_mol = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits_rna = list(kf_rna.split(unique_rnas))
        splits_mol = list(kf_mol.split(unique_mols))
        splits = list(zip(splits_rna, splits_mol))

    else:
        raise ValueError(f"Unsupported split_mode: {split_mode}")

    # 2. 执行 K-Fold 循环
    for fold, split_idx in enumerate(splits, 1):
        print(f"--- Generating Fold {fold}/{n_splits} (Mode: {split_mode}) ---")
        
        if split_mode == 'random':
            train_idx, test_idx = split_idx
            df_train_full = df_labels.iloc[train_idx]
            df_test = df_labels.iloc[test_idx]
            df_train, df_val = train_test_split(
                df_train_full, 
                test_size=val_size, 
                stratify=df_train_full[label_col], 
                random_state=seed
            )
            
        elif split_mode == 'cold_rna':
            train_idx, test_idx = split_idx
            train_val_rnas = unique_rnas[train_idx]
            test_rnas = unique_rnas[test_idx]
            tra_rnas, val_rnas = train_test_split(train_val_rnas, test_size=val_size, random_state=seed)
            df_train = df_labels[df_labels['RNA_ID'].isin(tra_rnas)]
            df_val = df_labels[df_labels['RNA_ID'].isin(val_rnas)]
            df_test = df_labels[df_labels['RNA_ID'].isin(test_rnas)]
            
        elif split_mode == 'cold_drug':
            train_idx, test_idx = split_idx
            train_val_mols = unique_mols[train_idx]
            test_mols = unique_mols[test_idx]
            tra_mols, val_mols = train_test_split(train_val_mols, test_size=val_size, random_state=seed)
            df_train = df_labels[df_labels[mol_id_col].isin(tra_mols)]
            df_val = df_labels[df_labels[mol_id_col].isin(val_mols)]
            df_test = df_labels[df_labels[mol_id_col].isin(test_mols)]

        elif split_mode == 'cold_both':
            (rna_tra_idx, rna_tes_idx), (mol_tra_idx, mol_tes_idx) = split_idx
            test_rnas = unique_rnas[rna_tes_idx]
            test_mols = unique_mols[mol_tes_idx]
            tv_rnas = unique_rnas[rna_tra_idx]
            tv_mols = unique_mols[mol_tra_idx]
            tra_rnas, val_rnas = train_test_split(tv_rnas, test_size=val_size, random_state=seed)
            tra_mols, val_mols = train_test_split(tv_mols, test_size=val_size, random_state=seed)
            
            df_train = df_labels[df_labels['RNA_ID'].isin(tra_rnas) & df_labels[mol_id_col].isin(tra_mols)]
            df_val = df_labels[df_labels['RNA_ID'].isin(val_rnas) & df_labels[mol_id_col].isin(val_mols)]
            df_test = df_labels[df_labels['RNA_ID'].isin(test_rnas) & df_labels[mol_id_col].isin(test_mols)]

        # 3. 合并特征数据并保存
        processed_dfs = []
        for df_sub, tvt_type in zip([df_train, df_val, df_test], ['tra', 'val', 'tes']):
            df_sub = df_sub.merge(df_rnas, left_on='RNA_ID', right_index=True)
            df_sub = df_sub.merge(df_Molecules, on=mol_id_col, how='left')
            smi_col = 'SMILES' if 'SMILES' in df_sub.columns else 'Canonical_SMILES'
            df_sub.dropna(subset=[smi_col], inplace=True)
            norm_path = os.path.normpath(dataset_path)
            clean_name = f"{norm_path.split(os.sep)[-1]}"
            out_path = f'data/processed/{clean_name}_fold{fold}_{tvt_type}_{split_mode}.csv'
            df_sub.to_csv(out_path, index=False)
            processed_dfs.append(df_sub)
        
        all_folds_data.append(processed_dfs)

    return all_folds_data

def trans_multimodal_offline(
    dataset_path, df_data, tvt_type, fold, args,
    mol_vec_map, rna_vec_map, smile_graph,
    split_mode='random'
):
    # === CRITICAL FIX: 统一参数对象为字典格式 ===
    args_dict = vars(args) if not isinstance(args, dict) else args
    
    norm_path = os.path.normpath(dataset_path)
    dataset_name = (
        f"{norm_path.split(os.sep)[-2]}_{norm_path.split(os.sep)[-1]}"
        if len(norm_path.split(os.sep)) >= 2
        else os.path.basename(dataset_path)
    )

    rna_seq_list = list(df_data['1D Sequence'])
    rna_id_list  = list(df_data['RNA_ID'])
    dot_list     = list(df_data['Dot bracket'])
    drug_smi     = list(df_data['SMILES'])
    drug_id_list = list(df_data['Small molecule_ID'])
    Y = np.asarray(df_data['label'])

    # 1. 计算 k-mer 特征
    kmer_dict = get_kmer_dict(args_dict['k_value'])
    k_mer_features = [
        get_kmer_index(seq, kmer_dict, args_dict['k_value'], args_dict['max_seq_len'])
        for seq in rna_seq_list
    ]

    data_list = []

    for i in tqdm(range(len(df_data)), desc=f"Fold {fold} {tvt_type}"):
        smiles = drug_smi[i]
        mid, rid = drug_id_list[i], rna_id_list[i]

        if smiles not in smile_graph:
            continue

        # ===== 模态 1: Drug Graph (原子级) =====
        _, atom_feat, edge_index = smile_graph[smiles]
        drug_data = Data(
            x=torch.tensor(atom_feat, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t()
                if len(edge_index) > 0 else torch.zeros((2, 0), dtype=torch.long)
        )

        # ===== 模态 2: RNA 1D (Sequence) =====
        rna_seq_data = torch.tensor(k_mer_features[i], dtype=torch.long)

        # ===== 模态 3: RNA 2D (Secondary Structure) =====
        seq = rna_seq_list[i]
        dot = dot_list[i]
        L = min(len(seq), len(dot), args_dict['max_seq_len'])
        r2d_x, r2d_ei, r2d_et = rna2D_from_dot(seq[:L], dot[:L])
        rna_2d_data = Data(x=r2d_x, edge_index=r2d_ei, edge_type=r2d_et)

        # ===== 模态 4: RNA 3D 多尺度 =====
        atom_graph, residue_graph = rna3D_from_pdb_cached(
            rid, 
            os.path.join(dataset_path, args_dict['pdb_dir']),
            dataset_path
        )
        
        has_3d = atom_graph is not None

        if not has_3d:
            # 缺失 3D 数据时的占位符处理，使用 args_dict 访问键值
            atom_graph = Data(
                x=torch.zeros((1, args_dict['node_dim_3d'])), 
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, args_dict['edge_dim_3d'])) 
            )
            residue_graph = Data(
                x=torch.zeros((1, 9)), 
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, 1))
            )

        # ===== 整合并保存所有模态字段 =====
        data = Data(
            drug=drug_data,
            rna_sequence=rna_seq_data,
            rna_2d=rna_2d_data,
            rna_3d=atom_graph,
            rna_3d_res=residue_graph,
            mol_sem_vec=mol_vec_map.get(mid, torch.zeros(768)),
            rna_sem_vec=rna_vec_map.get(rid, torch.zeros(768)),
            has_3d=torch.tensor([1.0 if has_3d else 0.0]),
            y=torch.tensor([Y[i]], dtype=torch.float)
        )

        data_list.append(data)

    dataset_obj = TestbedDataset(
        root='data',
        dataset=f'{dataset_name}_fold{fold}_{tvt_type}_{split_mode}',
        data_list=data_list
    )

    return dataset_obj

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='data/1D/RSID')
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--k_value', type=int, default=3)
    parser.add_argument('--max_seq_len', type=int, default=267)
    parser.add_argument('--pdb_dir', type=str, default='PDB')
    parser.add_argument('--split_mode', type=str, default='random', 
                        choices=['random', 'cold_rna', 'cold_drug', 'cold_both'])
    parser.add_argument('--bert_path', type=str, 
                        default='pretrained_models/biobert-base-cased-v1.2')
    parser.add_argument('--node_dim_3d', type=int, default=9)
    parser.add_argument('--edge_dim_3d', type=int, default=1)
    
    args = parser.parse_args()    

    # 1. 划分数据集
    all_folds = read_raw_data(args.dataset_path, args.n_splits, args.seed, args.val_size, args.split_mode)
    
    # 2. 预提取全量语义特征
    df_m = pd.read_excel(os.path.join(args.dataset_path, "Molecule.xlsx"))
    df_r = pd.read_excel(os.path.join(args.dataset_path, "RNA.xlsx"))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m_ids_col = 'Small molecule_ID' if 'Small molecule_ID' in df_m.columns else 'Molecule_ID'
    m_ids = df_m[m_ids_col].tolist()
    m_smis_col = 'SMILES' if 'SMILES' in df_m.columns else 'Canonical_SMILES'
    m_smis = df_m[m_smis_col].tolist()
    mol_vecs = get_bert_embeddings_batch(m_smis, args.bert_path, device)
    mol_vec_map = dict(zip(m_ids, mol_vecs))

    rna_ids = df_r['RNA_ID'].tolist()
    rna_seqs = df_r['1D Sequence'].tolist()
    rna_vecs = get_bert_embeddings_batch(rna_seqs, args.bert_path, device)
    rna_vec_map = dict(zip(rna_ids, rna_vecs))

    # 3. 预加载分子图
    smile_graph = get_smiles_graph_cached(args.dataset_path)

    # 4. 转换并保存 PyG 数据集
    for fold, (df_tra, df_val, df_tes) in enumerate(all_folds, 1):
        for df, tvt in zip([df_tra, df_val, df_tes], ['tra', 'val', 'tes']):
            trans_multimodal_offline(
                args.dataset_path, df, tvt, fold, args, 
                mol_vec_map, rna_vec_map, smile_graph, args.split_mode
            )