import argparse
import gc
import os
import random
import time

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from create_data import get_bert_embeddings_batch, get_smiles_graph_cached, read_raw_data, trans_multimodal_offline
from model import InteractionContrastiveLoss, MultiModalModel
from utils import TestbedDataset, get_metrics, set_seed


# =========================================================
# Data preparation
# =========================================================

def _dataset_name(dataset):
    norm_path = os.path.normpath(dataset)
    parts = norm_path.split(os.sep)
    return f"{parts[-2]}_{parts[-1]}" if len(parts) >= 2 else parts[-1]


def _select_semantic_text(dataframe, preferred_column, fallback_column):
    if preferred_column in dataframe.columns:
        return dataframe[preferred_column].fillna(dataframe[fallback_column]).astype(str).tolist()
    return dataframe[fallback_column].astype(str).tolist()


def check_and_prepare_data(args):
    dataset_name = _dataset_name(args["dataset"])
    sample_file = os.path.join("data", "processed", f"{dataset_name}_fold1_tra.pt")

    if os.path.exists(sample_file):
        print("[Data] Found preprocessed files; loading them directly.")
        return

    print(f"\n[Data] Preprocessed file not found: {sample_file}")
    print(">>> Starting offline preprocessing...")

    dataset_full_path = os.path.join("data", args["dataset"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_folds = read_raw_data(
        dataset_full_path,
        n_splits=args["n_splits"],
        seed=args["seed"],
        val_size=args["val_size"],
    )

    print(">>> Extracting semantic embeddings...")

    df_mol = pd.read_excel(os.path.join(dataset_full_path, "Molecule.xlsx"))
    df_rna = pd.read_excel(os.path.join(dataset_full_path, "RNA.xlsx"))

    mol_id_col = "Small molecule_ID" if "Small molecule_ID" in df_mol.columns else "Molecule_ID"
    smiles_col = "SMILES" if "SMILES" in df_mol.columns else "Canonical_SMILES"

    mol_texts = _select_semantic_text(df_mol, "Small molecule information", smiles_col)
    rna_texts = _select_semantic_text(df_rna, "RNA information", "1D Sequence")

    mol_vecs = get_bert_embeddings_batch(mol_texts, args["mol_semantic_model"], device)
    rna_vecs = get_bert_embeddings_batch(rna_texts, args["rna_semantic_model"], device)

    mol_map = dict(zip(df_mol[mol_id_col].tolist(), mol_vecs))
    rna_map = dict(zip(df_rna["RNA_ID"].tolist(), rna_vecs))
    smile_graph = get_smiles_graph_cached(dataset_full_path)

    for fold, (df_train, df_val, df_test) in enumerate(all_folds, 1):
        for df_split, tvt in zip([df_train, df_val, df_test], ["tra", "val", "tes"]):
            trans_multimodal_offline(
                dataset_full_path,
                df_split,
                tvt,
                fold,
                args,
                mol_map,
                rna_map,
                smile_graph,
            )

    print(">>> Offline preprocessing completed.\n")


# =========================================================
# Reproducibility and data loading
# =========================================================

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def compute_pos_weight(loader):
    labels = np.asarray([data.y.item() for data in loader.dataset])
    num_neg = (labels == 0).sum()
    num_pos = (labels == 1).sum()
    return torch.tensor([(num_neg + 1e-7) / (num_pos + 1e-7)], dtype=torch.float32)


def load_single_fold_dataloader(args, fold):
    dataset_name = _dataset_name(args["dataset"])

    generator = torch.Generator()
    generator.manual_seed(args["seed"])

    loaders = []
    for tvt in ["tra", "val", "tes"]:
        dataset_key = f"{dataset_name}_fold{fold}_{tvt}"
        dataset = TestbedDataset(root="data", dataset=dataset_key)
        loader = DataLoader(
            dataset,
            batch_size=args["batch_size"],
            shuffle=(tvt == "tra"),
            generator=generator,
            worker_init_fn=seed_worker,
        )
        loaders.append(loader)

    return tuple(loaders)


# =========================================================
# Training and evaluation
# =========================================================

def train_one_epoch(model, loader, loss_fn, optimizer, device, args):
    model.train()
    total_loss, y_true, y_pred = 0.0, [], []

    interaction_loss_fn = InteractionContrastiveLoss(temperature=args["contrastive_temp"])

    for data in loader:
        data = data.to(device)
        target = data.y.view(-1, 1).float()

        optimizer.zero_grad()
        output = model(data)
        logits = output["out"]

        interaction_loss = interaction_loss_fn(
            output["rna_feat"],
            output["drug_feat"],
            data.y.view(-1),
        )

        prediction_loss = loss_fn(logits, target)
        loss = (
            (1 - args["aux_weight_modal"]) * prediction_loss
            + args["aux_weight_modal"] * output["contrastive_loss"]
            + args["aux_weight_interaction"] * interaction_loss
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * target.size(0)
        y_pred.extend(torch.sigmoid(logits).detach().cpu().view(-1).tolist())
        y_true.extend(data.y.view(-1).cpu().tolist())

    return round(total_loss / len(y_true), 5), get_metrics(y_true, y_pred)


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, y_true, y_pred = 0.0, [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            target = data.y.view(-1, 1).float()

            logits = model(data)["out"]
            loss = loss_fn(logits, target)

            total_loss += loss.item() * target.size(0)
            y_pred.extend(torch.sigmoid(logits).cpu().view(-1).tolist())
            y_true.extend(data.y.view(-1).cpu().tolist())

    return round(total_loss / len(y_true), 5), get_metrics(y_true, y_pred)


# =========================================================
# Single-fold training
# =========================================================

def run_single_fold(fold, args, device):
    print(f"\n===== Fold {fold} / {args['n_splits']} =====")

    loader_train, loader_val, loader_test = load_single_fold_dataloader(args, fold)

    model = MultiModalModel(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=args["scheduler_patience"],
        factor=args["scheduler_factor"],
    )

    pos_weight = compute_pos_weight(loader_train).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)

    dataset_name = _dataset_name(args["dataset"])
    best_auc, best_epoch, best_test_metrics = -float("inf"), -1, None
    result_metrics = np.zeros((args["epochs"], 1 + 7 * 3))
    early_stop_counter = 0

    for epoch in range(args["epochs"]):
        start_time = time.time()

        train_loss, train_metrics = train_one_epoch(model, loader_train, loss_fn, optimizer, device, args)
        val_loss, val_metrics = evaluate(model, loader_val, loss_fn, device)
        test_loss, test_metrics = evaluate(model, loader_test, loss_fn, device)
        scheduler.step(val_loss)

        elapsed_minutes = round((time.time() - start_time) / 60, 1)

        print(f"--- epoch:{epoch:03d} | elapsed: {elapsed_minutes}m ---")
        print(f"Tra Loss: {train_loss:.5f} | Metrics: {train_metrics}")
        print(f"Val Loss: {val_loss:.5f} | Metrics: {val_metrics}")
        print(f"Tes Loss: {test_loss:.5f} | Metrics: {test_metrics}")

        result_metrics[epoch, 0] = epoch
        for i in range(7):
            result_metrics[epoch, 3 * i + 1] = train_metrics[i]
            result_metrics[epoch, 3 * i + 2] = val_metrics[i]
            result_metrics[epoch, 3 * i + 3] = test_metrics[i]

        if val_metrics[0] > best_auc:
            best_auc = val_metrics[0]
            best_epoch = epoch
            best_test_metrics = test_metrics
            early_stop_counter = 0

            os.makedirs("model", exist_ok=True)
            model_path = os.path.join("model", f"{dataset_name}_fold{fold}.pt")
            torch.save(model.state_dict(), model_path)
            print(f">>> Best Tes updated: {test_metrics}")
        else:
            early_stop_counter += 1
            print(f">>> No improvement for {early_stop_counter}/{args['early_stop_patience']}")

        print("-" * 30)

        if early_stop_counter >= args["early_stop_patience"]:
            print(
                f"\n[Early Stopping] Validation AUC has not improved for "
                f"{args['early_stop_patience']} epochs."
            )
            break

    print("\n" + "=" * 50)
    print(f"Fold {fold} training completed!")
    print(f"Best Epoch: {best_epoch:03d}")
    print(f"Corresponding test metrics: {best_test_metrics}")
    print("=" * 50 + "\n")

    actual_epochs = epoch + 1
    result_metrics = result_metrics[:actual_epochs, :]

    metrics = ["AUC", "AUPR", "F1", "Acc", "Rec", "Spec", "Prec"]
    partitions = ["Tra", "Val", "Tes"]
    columns = ["Epoch"] + [f"{partition}_{metric}" for metric in metrics for partition in partitions]

    os.makedirs("result", exist_ok=True)
    time_str = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    pd.DataFrame(result_metrics, columns=columns).to_csv(
        os.path.join("result", f"result_{dataset_name}_fold{fold}_{time_str}.csv"),
        index=False,
    )


# =========================================================
# Five-fold training
# =========================================================

def TraValTes(args):
    device = args["device"]
    check_and_prepare_data(args)

    for fold in range(1, args["n_splits"] + 1):
        run_single_fold(fold, args, device)

        print(f">>> Cleaning resources after Fold {fold}...")
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f">>> GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f">>> GPU Memory Reserved:  {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

        print("=" * 50 + "\n")
        time.sleep(2)


# =========================================================
# Arguments
# =========================================================

def get_args():
    parser = argparse.ArgumentParser()

    # Data / model
    parser.add_argument("--dataset", default="3D/alphafold3")
    parser.add_argument("--model_name", default="MultiModalModel")
    parser.add_argument("--pdb_dir", type=str, default="PDB")
    parser.add_argument("--mol_semantic_model", type=str, default="pretrained_models/biobert-base-cased-v1.2")
    parser.add_argument("--rna_semantic_model", type=str, default="pretrained_models/biobert-base-cased-v1.2")

    # Training
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--scheduler_patience", type=int, default=1)
    parser.add_argument("--scheduler_factor", type=float, default=0.7)
    parser.add_argument("--early_stop_patience", type=int, default=20)
    parser.add_argument("--n_splits", type=int, default=5)

    # Model architecture
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--nhead", type=int, default=32)
    parser.add_argument("--transformer_encoder_layer", type=int, default=1)
    parser.add_argument("--k_value", type=int, default=3)
    parser.add_argument("--max_seq_len", type=int, default=267)
    parser.add_argument("--dim_atom", type=int, default=78)
    parser.add_argument("--n_output", type=int, default=1)
    parser.add_argument("--num_base", type=int, default=4)
    parser.add_argument("--num_features_rna", type=int, default=5)
    parser.add_argument("--num_relations", type=int, default=9)
    parser.add_argument("--node_dim_3d", type=int, default=9)
    parser.add_argument("--edge_dim_3d", type=int, default=1)

    # Auxiliary objectives
    parser.add_argument("--contrastive_temp", type=float, default=0.1)
    parser.add_argument("--aux_weight_modal", type=float, default=0.25)
    parser.add_argument("--aux_weight_interaction", type=float, default=0.1)

    args, _ = parser.parse_known_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return vars(args)


if __name__ == "__main__":
    args = get_args()
    set_seed(args["seed"])
    TraValTes(args)
