import os
import gc  # 垃圾回收模块

# =========================================================
# CRITICAL FIX: 环境变量设置
# =========================================================
# 1. 确保 GNN scatter 确定性
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# 2. 抑制 HuggingFace 警告
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# 3. [关键] 限制 PyTorch 显存碎片，防止长时间运行后的 OOM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from datetime import datetime
import random

# 导入自定义模块
from model import MultiModalModel, InteractionContrastiveLoss
from create_data import (
    read_raw_data, 
    trans_multimodal_offline, 
    get_bert_embeddings_batch, 
    get_smiles_graph_cached
)
from utils import set_seed, get_metrics, TestbedDataset

# ==========================
# 1. 数据检查与预处理 (保持不变)
# ==========================

def check_and_prepare_data(args):
    norm_path = os.path.normpath(args['dataset'])
    path_parts = norm_path.split(os.sep)
    dataset_name = f"{path_parts[-2]}_{path_parts[-1]}" if len(path_parts) >= 2 else path_parts[-1]
    
    sample_file = os.path.join('data', 'processed', f"{dataset_name}_fold1_tra_{args['split_mode']}.pt")
    
    if not os.path.exists(sample_file):
        print(f"\n[数据检查] 未找到模式为 {args['split_mode']} 的预处理文件: {sample_file}")
        print(">>> 正在启动自动离线预处理程序...")
        
        dataset_full_path = os.path.join('data', args['dataset'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        all_folds = read_raw_data(
            dataset_full_path, 
            args['n_splits'], 
            args['seed'], 
            args['val_size'], 
            args['split_mode']
        )
        
        print(">>> 正在批量提取 BERT 语义向量...")
        df_mol_all = pd.read_excel(os.path.join(dataset_full_path, "Molecule.xlsx"))
        df_rna_all = pd.read_excel(os.path.join(dataset_full_path, "RNA.xlsx"))
        
        mol_map = dict(zip(df_mol_all['Small molecule_ID'], 
                           get_bert_embeddings_batch(df_mol_all['Small molecule information'].fillna(df_mol_all['SMILES']).tolist(), 
                                                     args['mol_semantic_model'], device)))
        rna_map = dict(zip(df_rna_all['RNA_ID'], 
                           get_bert_embeddings_batch(df_rna_all['RNA information'].fillna(df_rna_all['1D Sequence']).tolist(), 
                                                     args['rna_semantic_model'], device)))
        
        smile_graph = get_smiles_graph_cached(dataset_full_path)
        
        for fold, (df_tra, df_val, df_tes) in enumerate(all_folds, 1):
            for df, tvt in zip([df_tra, df_val, df_tes], ['tra', 'val', 'tes']):
                trans_multimodal_offline(
                    dataset_full_path, df, tvt, fold, 
                    args, mol_map, rna_map, smile_graph, args['split_mode']
                )
        print(f">>> 模式为 {args['split_mode']} 的自动化数据预处理已完成！\n")
    else:
        print(f"[数据检查] 检测到已存在模式为 {args['split_mode']} 的文件，直接进入加载阶段。")

# ==========================
# 2. 工具函数 (Worker & Weight)
# ==========================

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def compute_pos_weight(loader):
    labels = [data.y.item() for data in loader.dataset]
    num_neg = (np.array(labels) == 0).sum()
    num_pos = (np.array(labels) == 1).sum()
    return torch.tensor([(num_neg + 1e-7) / (num_pos + 1e-7)], dtype=torch.float32)

# ==========================
# 3. 按需数据加载 (优化内存)
# ==========================

def load_single_fold_dataloader(args, fold):
    """
    只加载当前这一折的数据，避免一次性加载所有折导致内存溢出
    """
    norm_path = os.path.normpath(args['dataset'])
    path_parts = norm_path.split(os.sep)
    dataset_name = f"{path_parts[-2]}_{path_parts[-1]}" if len(path_parts) >= 2 else path_parts[-1]
    
    g = torch.Generator()
    g.manual_seed(args['seed'])

    fold_loaders = []
    # 依次加载 tra, val, tes
    for tvt in ['tra', 'val', 'tes']:
        dataset_key = f"{dataset_name}_fold{fold}_{tvt}_{args['split_mode']}"
        ds = TestbedDataset(root='data', dataset=dataset_key)
        
        loader = DataLoader(
            ds,
            batch_size=args['batch_size'],
            shuffle=(tvt == 'tra'),
            generator=g,
            worker_init_fn=seed_worker
        )
        fold_loaders.append(loader)
    
    return tuple(fold_loaders)

# ==========================
# 4. 训练与评估核心逻辑
# ==========================

def train_one_epoch(model, loader, loss_fn, optimizer, device, args):
    model.train()
    total_loss, y_true, y_pred = 0, [], []
    interaction_loss_fn = InteractionContrastiveLoss(temperature=args['contrastive_temp'])

    for data in loader:
        data = data.to(device)
        target = data.y.view(-1, 1).float()
        optimizer.zero_grad()
        
        output = model(data)
        logits = output['out']
        intra_loss = interaction_loss_fn(output['rna_feat'], output['drug_feat'], data.y.view(-1))
        
        loss = loss_fn(logits, target)
        if args['use_semantic']:
            loss = (1 - args['aux_weight_modal']) * loss + args['aux_weight_modal'] * output['contrastive_loss']
        
        loss = loss + args['aux_weight_interaction'] * intra_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * target.size(0)
        y_pred += torch.sigmoid(logits).detach().cpu().view(-1).tolist()
        y_true += data.y.view(-1).cpu().tolist()
    return round(total_loss / len(y_true), 5), get_metrics(y_true, y_pred)

def evaluate(model, loader, loss_fn, device, args):
    model.eval()
    total_loss, y_true, y_pred = 0, [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            target = data.y.view(-1, 1).float().to(device)
            output = model(data)
            logits = output['out']
            loss = loss_fn(logits, target)

            total_loss += loss.item() * target.size(0)
            y_pred += torch.sigmoid(logits).cpu().view(-1).tolist()
            y_true += data.y.view(-1).cpu().tolist()

    return round(total_loss / len(y_true), 5), get_metrics(y_true, y_pred)

# ==========================
# 5. 单折运行逻辑 (封装)
# ==========================

def run_single_fold(fold, args, device):
    """
    执行单折训练。
    函数结束后，model, optimizer, loaders 等局部变量会自动被 Python 销毁。
    这是避免显存泄漏最有效的方法。
    """
    print(f"\n===== Fold {fold} / {args['n_splits']} (Split: {args['split_mode']}) =====")
    
    # 1. 动态加载数据 (局部变量)
    loader_tra, loader_val, loader_tes = load_single_fold_dataloader(args, fold)
    
    # 2. 初始化模型与优化器 (局部变量)
    model = MultiModalModel(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=args['scheduler_patience'], factor=args['scheduler_factor']
    )
    
    pos_weight = compute_pos_weight(loader_tra).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)

    # 3. 准备记录变量
    norm_path = os.path.normpath(args['dataset'])
    path_parts = norm_path.split(os.sep)
    dataset_name = f"{path_parts[-2]}_{path_parts[-1]}" if len(path_parts) >= 2 else path_parts[-1]

    best_auc = 0
    best_epoch = -1
    best_tes_metrics = None
    result_metrics = np.zeros((args['epochs'], 1 + 7 * 3))
    early_stop_counter = 0

    # 4. Epoch 循环
    for epoch in range(args['epochs']):
        t0 = time.time()
        loss_tra, met_tra = train_one_epoch(model, loader_tra, loss_fn, optimizer, device, args)
        loss_val, met_val = evaluate(model, loader_val, loss_fn, device, args)
        loss_tes, met_tes = evaluate(model, loader_tes, loss_fn, device, args)
        scheduler.step(loss_val)

        t1 = round((time.time() - t0) / 60, 1)
        
        print(f"--- epoch:{epoch:03d} | elapsed: {t1}m ---")
        print(f"Tra Loss: {loss_tra:.5f} | Metrics: {met_tra}")
        print(f"Val Loss: {loss_val:.5f} | Metrics: {met_val}")
        print(f"Tes Loss: {loss_tes:.5f} | Metrics: {met_tes}")

        # 记录数据
        result_metrics[epoch, 0] = epoch
        for i in range(7):
            result_metrics[epoch, 3 * i + 1] = met_tra[i]
            result_metrics[epoch, 3 * i + 2] = met_val[i]
            result_metrics[epoch, 3 * i + 3] = met_tes[i]

        # 模型保存与早停
        if met_val[0] > best_auc:
            best_auc = met_val[0]
            best_epoch = epoch
            best_tes_metrics = met_tes
            early_stop_counter = 0
            os.makedirs("model", exist_ok=True)
            model_path = f"model/{dataset_name}_{args['split_mode']}_fold{fold}.pt"
            torch.save(model.state_dict(), model_path)
            print(f">>> Best Tes updated: {met_tes}")
        else:
            early_stop_counter += 1
            print(f">>> No improvement for {early_stop_counter}/{args['early_stop_patience']}")
            
        print("-" * 30)
        
        if early_stop_counter >= args['early_stop_patience']:
            print(f"\n[Early Stopping] Validation AUC has not improved for {args['early_stop_patience']} epochs.")
            break

    # 5. 训练结束总结与保存结果
    print(f"\n" + "="*50)
    print(f"Fold {fold} 训练完成!")
    print(f"最佳 Epoch: {best_epoch:03d}")
    print(f"对应测试集指标: {best_tes_metrics}")
    print("="*50 + "\n")

    actual_epochs = epoch + 1
    result_metrics_truncated = result_metrics[:actual_epochs, :]
    cols = ['Epoch'] + [f"{p}_{m}" for m in ['AUC','AUPR','F1','Acc','Rec','Spec','Prec'] for p in ['Tra','Val','Tes']]
    df = pd.DataFrame(result_metrics_truncated, columns=cols)
    os.makedirs('result', exist_ok=True)
    time_str = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())
    df.to_csv(f"result/result_{dataset_name}_{args['split_mode']}_fold{fold}_{time_str}.csv", index=False)
    
    # 函数结束，返回 None，Python 自动回收内部变量
    return

# ==========================
# 6. 主调度器 (内存管理核心)
# ==========================

def TraValTes(args):
    device = args['device']
    
    # 1. 预处理检查
    check_and_prepare_data(args)
    
    # 2. 循环执行每一折
    for fold in range(1, args['n_splits'] + 1):
        
        # === 调用封装函数执行单折训练 ===
        # 所有模型、数据、优化器都在这个函数内部创建和销毁
        run_single_fold(fold, args, device)
        
        # === 关键：内存深度清理 ===
        print(f">>> 正在清理 Fold {fold} 的残留显存...")
        
        # 强制 Python 垃圾回收 (清理 CPU 内存中的对象引用)
        gc.collect()
        
        # 强制 PyTorch 释放缓存 (清理 GPU 显存)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize() # 等待所有 CUDA 操作完成
            
            # 打印显存状态确认清理效果
            print(f">>> GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f">>> GPU Memory Reserved:  {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        
        print("="*50 + "\n")
        
        # 等待一小会儿确保 OS 回收资源
        time.sleep(2)

# ==========================
# 7. 参数配置
# ==========================

def get_args():
    parser = argparse.ArgumentParser()
    # 路径与名称
    parser.add_argument('--dataset', default='3D/alphafold3')
    parser.add_argument('--model_name', default='MultiModalModel')
    # 训练配置
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--scheduler_patience', type=int, default=1)  
    parser.add_argument('--scheduler_factor', type=float, default=0.7)
    parser.add_argument('--early_stop_patience', type=int, default=20)

    # 模型架构
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--nhead', type=int, default=32)
    parser.add_argument('--transformer_encoder_layer', type=int, default=1)
    parser.add_argument('--k_value', type=int, default=3)
    parser.add_argument('--max_seq_len', type=int, default=267)
    parser.add_argument('--dim_atom', type=int, default=78)
    parser.add_argument('--n_output', type=int, default=1)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--num_base', type=int, default=4)
    parser.add_argument('--num_features_rna', type=int, default=5)
    parser.add_argument('--num_relations', type=int, default=9)
    parser.add_argument('--node_dim_3d', type=int, default=9)
    parser.add_argument('--edge_dim_3d', type=int, default=1)
    # 损失控制
    parser.add_argument('--contrastive_temp', type=float, default=0.1)
    parser.add_argument('--aux_weight_modal', type=float, default=0.25)
    parser.add_argument('--aux_weight_interaction', type=float, default=0.1)
    
    # 消融开关
    parser.add_argument('--gin_single_scale', type=int, default=0)
    parser.add_argument('--wo_dynamic_weight', type=int, default=0)
    parser.add_argument('--wo_cross_attn', type=int, default=0)
    parser.add_argument('--use_pos_encoding', type=int, default=1)
    parser.add_argument('--use_atom_3d', type=int, default=1)
    parser.add_argument('--use_residue_3d', type=int, default=1)
    parser.add_argument('--use_torsion', type=int, default=1)
    
    # 模态与划分
    parser.add_argument('--use_semantic', type=int, default=1)
    parser.add_argument('--use_rna_2d', type=int, default=1) 
    parser.add_argument('--use_rna_3d', type=int, default=1)
    parser.add_argument('--split_mode', type=str, default='random', 
                        choices=['random', 'cold_rna', 'cold_drug', 'cold_both'])
    parser.add_argument('--pdb_dir', type=str, default='PDB')
    parser.add_argument('--mol_semantic_model', type=str, default='pretrained_models/biobert-base-cased-v1.2')
    parser.add_argument('--rna_semantic_model', type=str, default='pretrained_models/biobert-base-cased-v1.2')
    
    # 兼容 Colab (sys.argv)
    args, unknown = parser.parse_known_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return vars(args)

if __name__ == '__main__':
    args = get_args()
    set_seed(args['seed'])
    TraValTes(args)