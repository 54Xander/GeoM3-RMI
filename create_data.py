import os

import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.data import Data
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from utils import (
    TestbedDataset,
    get_kmer_dict,
    get_kmer_index,
    get_smiles_graph_cached,
    rna2D_from_dot,
    rna3D_from_pdb_cached,
)


RDLogger.DisableLog("rdApp.*")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.makedirs("data/processed/cache", exist_ok=True)


# =========================================================
# Semantic embeddings
# =========================================================

def get_bert_embeddings_batch(text_list, model_name, device, batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embeddings = []
    print(f"Extracting BERT embeddings (Batch size: {batch_size})...")

    with torch.no_grad():
        for i in tqdm(range(0, len(text_list), batch_size)):
            batch_texts = [str(text) for text in text_list[i:i + batch_size]]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)
            embeddings = model(**inputs).last_hidden_state[:, 0, :].cpu()
            all_embeddings.append(embeddings)

    return torch.cat(all_embeddings, dim=0)


# =========================================================
# Random stratified five-fold split
# =========================================================

def read_raw_data(dataset_path, n_splits=5, seed=42, val_size=0.1):
    df_molecules = pd.read_excel(os.path.join(dataset_path, "Molecule.xlsx"))
    df_rnas = pd.read_excel(os.path.join(dataset_path, "RNA.xlsx")).set_index("RNA_ID")
    df_labels = pd.read_excel(os.path.join(dataset_path, "RNA-Molecule.xlsx"))

    label_col = "label" if "label" in df_labels.columns else "Label"
    mol_id_col = "Small molecule_ID" if "Small molecule_ID" in df_labels.columns else "Molecule_ID"

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_folds_data = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(df_labels, df_labels[label_col]), 1):
        print(f"--- Generating Fold {fold}/{n_splits} ---")

        df_train_full = df_labels.iloc[train_idx]
        df_test = df_labels.iloc[test_idx]
        df_train, df_val = train_test_split(
            df_train_full,
            test_size=val_size,
            stratify=df_train_full[label_col],
            random_state=seed,
        )

        processed_dfs = []

        for df_sub, tvt_type in zip([df_train, df_val, df_test], ["tra", "val", "tes"]):
            df_sub = df_sub.merge(df_rnas, left_on="RNA_ID", right_index=True)
            df_sub = df_sub.merge(df_molecules, on=mol_id_col, how="left")

            smiles_col = "SMILES" if "SMILES" in df_sub.columns else "Canonical_SMILES"
            df_sub = df_sub.dropna(subset=[smiles_col]).copy()

            if smiles_col != "SMILES" and "SMILES" not in df_sub.columns:
                df_sub["SMILES"] = df_sub[smiles_col]

            if mol_id_col != "Small molecule_ID" and "Small molecule_ID" not in df_sub.columns:
                df_sub["Small molecule_ID"] = df_sub[mol_id_col]

            clean_name = os.path.basename(os.path.normpath(dataset_path))
            out_path = os.path.join("data", "processed", f"{clean_name}_fold{fold}_{tvt_type}.csv")
            df_sub.to_csv(out_path, index=False)
            processed_dfs.append(df_sub)

        all_folds_data.append(processed_dfs)

    return all_folds_data


# =========================================================
# Multimodal offline preprocessing
# =========================================================

def trans_multimodal_offline(
    dataset_path,
    df_data,
    tvt_type,
    fold,
    args,
    mol_vec_map,
    rna_vec_map,
    smile_graph,
):
    args_dict = vars(args) if not isinstance(args, dict) else args

    norm_path = os.path.normpath(dataset_path)
    parts = norm_path.split(os.sep)
    dataset_name = f"{parts[-2]}_{parts[-1]}" if len(parts) >= 2 else os.path.basename(dataset_path)

    rna_seq_list = list(df_data["1D Sequence"])
    rna_id_list = list(df_data["RNA_ID"])
    dot_list = list(df_data["Dot bracket"])

    smiles_col = "SMILES" if "SMILES" in df_data.columns else "Canonical_SMILES"
    mol_id_col = "Small molecule_ID" if "Small molecule_ID" in df_data.columns else "Molecule_ID"
    label_col = "label" if "label" in df_data.columns else "Label"

    drug_smi = list(df_data[smiles_col])
    drug_id_list = list(df_data[mol_id_col])
    labels = np.asarray(df_data[label_col])

    kmer_dict = get_kmer_dict(args_dict["k_value"])
    kmer_features = [
        get_kmer_index(seq, kmer_dict, args_dict["k_value"], args_dict["max_seq_len"])
        for seq in rna_seq_list
    ]

    data_list = []

    for i in tqdm(range(len(df_data)), desc=f"Fold {fold} {tvt_type}"):
        smiles = drug_smi[i]
        mol_id = drug_id_list[i]
        rna_id = rna_id_list[i]

        if smiles not in smile_graph:
            continue

        # Small-molecule graph
        _, atom_feat, edge_index = smile_graph[smiles]
        drug_data = Data(
            x=torch.tensor(atom_feat, dtype=torch.float),
            edge_index=(
                torch.tensor(edge_index, dtype=torch.long).t()
                if len(edge_index) > 0
                else torch.zeros((2, 0), dtype=torch.long)
            ),
        )

        # RNA 1D
        rna_seq_data = torch.tensor(kmer_features[i], dtype=torch.long)

        # RNA 2D
        seq, dot = rna_seq_list[i], dot_list[i]
        length = min(len(seq), len(dot), args_dict["max_seq_len"])
        r2d_x, r2d_ei, r2d_et = rna2D_from_dot(seq[:length], dot[:length])
        rna_2d_data = Data(x=r2d_x, edge_index=r2d_ei, edge_type=r2d_et)

        # RNA dual-scale 3D
        atom_graph, residue_graph = rna3D_from_pdb_cached(
            rna_id,
            os.path.join(dataset_path, args_dict["pdb_dir"]),
            dataset_path,
        )
        has_3d = atom_graph is not None

        if not has_3d:
            atom_graph = Data(
                x=torch.zeros((1, args_dict["node_dim_3d"])),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, args_dict["edge_dim_3d"])),
            )
            residue_graph = Data(
                x=torch.zeros((1, 9)),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, 1)),
            )

        data = Data(
            drug=drug_data,
            rna_sequence=rna_seq_data,
            rna_2d=rna_2d_data,
            rna_3d=atom_graph,
            rna_3d_res=residue_graph,
            mol_sem_vec=mol_vec_map.get(mol_id, torch.zeros(768)),
            rna_sem_vec=rna_vec_map.get(rna_id, torch.zeros(768)),
            has_3d=torch.tensor([1.0 if has_3d else 0.0]),
            y=torch.tensor([labels[i]], dtype=torch.float),
        )
        data_list.append(data)

    return TestbedDataset(
        root="data",
        dataset=f"{dataset_name}_fold{fold}_{tvt_type}",
        data_list=data_list,
    )


def _select_semantic_text(dataframe, preferred_column, fallback_column):
    if preferred_column in dataframe.columns:
        return dataframe[preferred_column].fillna(dataframe[fallback_column]).astype(str).tolist()
    return dataframe[fallback_column].astype(str).tolist()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/3D/alphafold3")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k_value", type=int, default=3)
    parser.add_argument("--max_seq_len", type=int, default=267)
    parser.add_argument("--pdb_dir", type=str, default="PDB")
    parser.add_argument("--bert_path", type=str, default="pretrained_models/biobert-base-cased-v1.2")
    parser.add_argument("--node_dim_3d", type=int, default=9)
    parser.add_argument("--edge_dim_3d", type=int, default=1)
    args = parser.parse_args()

    all_folds = read_raw_data(args.dataset_path, args.n_splits, args.seed, args.val_size)

    df_mol = pd.read_excel(os.path.join(args.dataset_path, "Molecule.xlsx"))
    df_rna = pd.read_excel(os.path.join(args.dataset_path, "RNA.xlsx"))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mol_id_col = "Small molecule_ID" if "Small molecule_ID" in df_mol.columns else "Molecule_ID"
    smiles_col = "SMILES" if "SMILES" in df_mol.columns else "Canonical_SMILES"

    mol_ids = df_mol[mol_id_col].tolist()
    mol_texts = _select_semantic_text(df_mol, "Small molecule information", smiles_col)
    mol_vecs = get_bert_embeddings_batch(mol_texts, args.bert_path, device)
    mol_vec_map = dict(zip(mol_ids, mol_vecs))

    rna_ids = df_rna["RNA_ID"].tolist()
    rna_texts = _select_semantic_text(df_rna, "RNA information", "1D Sequence")
    rna_vecs = get_bert_embeddings_batch(rna_texts, args.bert_path, device)
    rna_vec_map = dict(zip(rna_ids, rna_vecs))

    smile_graph = get_smiles_graph_cached(args.dataset_path)

    for fold, (df_train, df_val, df_test) in enumerate(all_folds, 1):
        for df_split, tvt in zip([df_train, df_val, df_test], ["tra", "val", "tes"]):
            trans_multimodal_offline(
                args.dataset_path,
                df_split,
                tvt,
                fold,
                args,
                mol_vec_map,
                rna_vec_map,
                smile_graph,
            )
