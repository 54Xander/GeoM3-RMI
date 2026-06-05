import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, AttentionalAggregation, RGCNConv, global_mean_pool
from torch_geometric.data import Batch
import math
from torch_geometric.nn import MessagePassing

# --- 基础模块 ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class MyMHA(nn.Module):
    def __init__(self, args):
        super().__init__()
        embed_dim = args['embed_dim']
        num_heads = args['nhead']
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.prior_weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, x_q, x_k, x_v, prior_attn=None, padding_mask=None):
        B, T_q, D = x_q.shape
        T_kv, H, d_k = x_k.shape[1], self.num_heads, self.head_dim
        Q = self.q_proj(x_q).view(B, T_q, H, d_k).transpose(1, 2)
        K = self.k_proj(x_k).view(B, T_kv, H, d_k).transpose(1, 2)
        V = self.v_proj(x_v).view(B, T_kv, H, d_k).transpose(1, 2)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        if padding_mask is not None:
            mask = padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        if prior_attn is not None:
            w = torch.sigmoid(self.prior_weight)
            attn_weights = (1 - w) * attn_weights + w * prior_attn.unsqueeze(1)
        attn_output = torch.matmul(attn_weights, V).transpose(1, 2).contiguous().view(B, T_q, D)
        return self.out_proj(attn_output), attn_weights

def masked_mean_pooling(x, mask):
    mask = mask[:, :x.size(1)] 
    v = (~mask).unsqueeze(-1).float()
    return (x * v).sum(1) / v.sum(1).clamp(min=1)

# --- 特征支路 ---
class MultiScaleGIN(nn.Module):
    def __init__(self, dim_atom, dim_hid, dropout, gin_single_scale=False):
        super().__init__()
        self.gin_single_scale = gin_single_scale
        def mk_nn(i, o): return nn.Sequential(nn.Linear(i, o), nn.ReLU(), nn.Linear(o, o))
        self.conv1 = GINConv(mk_nn(dim_atom, dim_hid))
        self.conv2 = GINConv(mk_nn(dim_hid, dim_hid))
        self.conv3 = GINConv(mk_nn(dim_hid, dim_hid))
        self.bn1, self.bn2, self.bn3 = nn.LayerNorm(dim_hid), nn.LayerNorm(dim_hid), nn.LayerNorm(dim_hid)
        self.pool = AttentionalAggregation(nn.Sequential(nn.Linear(dim_hid, 1), nn.ReLU()))
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch):
        x1 = self.relu(self.bn1(self.conv1(x, edge_index)))
        if self.gin_single_scale:
            # 消融实验：仅使用第一层特征
            return self.pool(x1, batch), x1
        
        x2 = self.relu(self.bn2(self.conv2(x1, edge_index)))
        x3 = self.relu(self.bn3(self.conv3(x2, edge_index)))
        # 默认：多尺度拼接
        g = torch.cat([self.pool(x1, batch), self.pool(x2, batch), self.pool(x3, batch)], dim=1)
        return g, torch.cat([x1, x2, x3], dim=1)

class RNA_RGCN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = RGCNConv(args['num_features_rna'], args['embed_dim'], args['num_relations'])
        self.conv2 = RGCNConv(args['embed_dim'], args['embed_dim'], args['num_relations'])
        self.fc, self.relu, self.drop = nn.Linear(args['embed_dim'], args['embed_dim']), nn.ReLU(), nn.Dropout(args['dropout'])
    def forward(self, b):
        x = self.relu(self.conv1(b.x, b.edge_index, b.edge_type))
        x = global_mean_pool(self.conv2(x, b.edge_index, b.edge_type), b.batch)
        return self.drop(self.relu(self.fc(x)))

class GeometricGNN(MessagePassing):
    def __init__(self, h, nd=9, ed=1):
        super().__init__(aggr='add')
        self.mlp = nn.Sequential(nn.Linear(nd + ed, h), nn.ReLU(), nn.Linear(h, h))
        self.out = nn.Sequential(nn.Linear(h, h), nn.ReLU())
    def forward(self, x, ei, ea, b): return self.out(self.propagate(ei, x=x, edge_attr=ea, batch=b))
    def message(self, x_j, edge_attr): return self.mlp(torch.cat([x_j, edge_attr], dim=-1))

# --- 损失与融合 ---
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__(); self.t = temperature
    def forward(self, x1, x2):
        m = (x1.abs().sum(1) > 1e-6) & (x2.abs().sum(1) > 1e-6)
        if m.sum() < 2: return torch.tensor(0.0, requires_grad=True, device=x1.device)
        s = F.cosine_similarity(x1[m].unsqueeze(1), x2[m].unsqueeze(0), dim=-1) / self.t
        return F.cross_entropy(s, torch.arange(s.size(0), device=s.device))

class InteractionContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__(); self.t = temperature
    def forward(self, r, d, l):
        s = F.cosine_similarity(F.normalize(r, dim=1), F.normalize(d, dim=1)) / self.t
        p, n = l == 1, l == 0
        lp = (1 - s[p]).mean() if p.sum() > 0 else 0.0
        ln = (s[n] + 1).clamp(min=0).mean() if n.sum() > 0 else 0.0
        return lp + ln

class DynamicWeightNet(nn.Module):
    def __init__(self, i, n):
        super().__init__(); self.fc = nn.Linear(i, n)
    def forward(self, x): return self.fc(x)

class InterModalityAttention(nn.Module):
    def __init__(self, d, n):
        super().__init__(); self.fc = nn.Linear(d, 1)
    def forward(self, fs):
        h = torch.stack(fs, dim=1)
        w = F.softmax(self.fc(h).squeeze(-1), dim=-1)
        return torch.sum(h * w.unsqueeze(2), dim=1), w

# --- 主模型 ---
class MultiModalModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device, self.embed_dim = args['device'], args['embed_dim']
        self.use_rna_2d = args['use_rna_2d']
        self.use_rna_3d = args['use_rna_3d']
        self.use_semantic = args.get('use_semantic', 1)
        self.neg_inf = -1e9

        # === 核心消融开关 (完整保留) ===
        self.gin_single_scale = args.get('gin_single_scale', 0)
        self.wo_cross_attn = args.get('wo_cross_attn', 0)
        self.wo_dynamic_weight = args.get('wo_dynamic_weight', 0)
        self.use_pos_encoding = args.get('use_pos_encoding', 1)
        
        # === 3D 消融开关 ===
        self.use_atom_3d = args.get('use_atom_3d', 1)       # 消融：原子级 3D 控制
        self.use_residue_3d = args.get('use_residue_3d', 1) # 消融：残基级 3D 控制
        self.use_torsion = args.get('use_torsion', 1)       # 消融：扭转角控制

        # --- 小分子支路 (未删减) ---
        self.drug_gin = MultiScaleGIN(args['dim_atom'], args['embed_dim'], args['dropout'], gin_single_scale=self.gin_single_scale)
        drug_in_dim = args['embed_dim'] if self.gin_single_scale else 3 * args['embed_dim']
        self.drug_proj = nn.Linear(drug_in_dim, args['embed_dim'])
        
        # --- RNA 支路 (未删减) ---
        self.embedding_xt = nn.Embedding(args['num_base'] ** args['k_value'] + 1, args['embed_dim'], padding_idx=0)
        self.pos_encoder = PositionalEncoding(args['embed_dim'], args['max_seq_len'])
        self.rna_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=args['embed_dim'], nhead=args['nhead'], batch_first=True, dropout=args['dropout']), 
            num_layers=args['transformer_encoder_layer']
        )
        self.rna_fc = nn.Linear(args['embed_dim'], args['embed_dim'])
        self.rna_rgcn = RNA_RGCN(args)
        
        # --- 3D 支路 1: 原子级 (微观) ---
        self.rna_3d_gnn = GeometricGNN(args['embed_dim'], args['node_dim_3d'], args['edge_dim_3d'])

        # --- 3D 支路 2: 残基级 (宏观/多尺度新增) ---    
        if self.use_residue_3d:
            res_node_dim = 3 + (6 if self.use_torsion else 0)
            # 按照定义顺序：hidden_dim, node_dim, edge_dim
            self.res_3d_gnn = GeometricGNN(args['embed_dim'], res_node_dim, 1)

        # --- 语义支路 ---
        if self.use_semantic:
            self.mol_sem_proj = nn.Linear(768, args['embed_dim'])
            self.rna_sem_proj = nn.Linear(768, args['embed_dim'])

        # --- 动态权重模块 (更新 n_rna 计数) ---
        n_mol = 1 + self.use_semantic
        self.drug_dyn = DynamicWeightNet(n_mol * self.embed_dim, n_mol)
        
        # 计算 RNA 模态总数：1D(1) + 2D(if) + 原子级 3D(if) + 残基级 3D(if) + 语义(if)
        n_rna = (
            1
            + self.use_rna_2d
            + (self.use_rna_3d and self.use_atom_3d)
            + (self.use_rna_3d and self.use_residue_3d)
            + self.use_semantic
        )
        self.rna_dyn = DynamicWeightNet(n_rna * self.embed_dim, n_rna)
        
        # --- 交互层与输出 (未删减) ---
        if self.wo_cross_attn:
            self.fusion_concat = nn.Linear(args['embed_dim'] * 2, args['embed_dim'])
        else:
            self.MHA_drug_from_rna = MyMHA(args)
            self.MHA_rna_from_drug = MyMHA(args)
            self.inter_modality_attn = InterModalityAttention(self.embed_dim, 2)

        self.fc1 = nn.Linear(self.embed_dim, args['embed_dim'] * 2)
        self.fc2 = nn.Linear(args['embed_dim'] * 2, args['embed_dim'])
        self.out = nn.Linear(args['embed_dim'], args['n_output'])
        self.cl_loss = ContrastiveLoss(temperature=args.get('contrastive_temp', 0.1))

    def forward(self, data):
        B = data.y.size(0)

        # 1. 小分子特征提取
        drug_batch = Batch.from_data_list(data.drug).to(self.device)
        dg, _ = self.drug_gin(drug_batch.x, drug_batch.edge_index, drug_batch.batch)
        drug_g = self.drug_proj(dg)

        m_list, m_masks = [drug_g], [torch.ones(B, 1, device=self.device)]
        if self.use_semantic:
            m_sem_vec = data.mol_sem_vec.view(B, -1)
            m_sem = self.mol_sem_proj(m_sem_vec)
            m_list.append(m_sem)
            m_masks.append((m_sem_vec.abs().sum(1, keepdim=True) > 1e-6).float())

        if self.wo_dynamic_weight:
            drug_fused = torch.stack(m_list, dim=1).mean(dim=1)
        else:
            m_msk = torch.cat(m_masks, dim=1)
            mw = F.softmax(self.drug_dyn(torch.cat(m_list, dim=1)).masked_fill(m_msk == 0, self.neg_inf), dim=1)
            drug_fused = torch.sum(torch.stack(m_list, dim=1) * mw.unsqueeze(2), dim=1)

        # 2. RNA 特征提取
        rs = data.rna_sequence.view(B, -1).long()
        rm = (rs == 0)
        rna_emb = self.embedding_xt(rs)
        if self.use_pos_encoding:
            rna_emb = self.pos_encoder(rna_emb)
        ro = self.rna_transformer(rna_emb, src_key_padding_mask=rm)
        r1d = masked_mean_pooling(self.rna_fc(ro), rm)

        r_list, r_masks = [r1d], [torch.ones(B, 1, device=self.device)]
        
        # RNA 2D
        if self.use_rna_2d:
            r2d_batch = Batch.from_data_list(data.rna_2d).to(self.device)
            r2d = self.rna_rgcn(r2d_batch) 
            r_list.append(r2d); r_masks.append(torch.ones(B, 1, device=self.device))
            
        # RNA 3D - 原子级 (Atom-level)
        r3d_atom = torch.zeros(B, self.embed_dim, device=self.device)
        if self.use_rna_3d and self.use_atom_3d:
            r3d_batch = Batch.from_data_list(data.rna_3d).to(self.device)
            if hasattr(r3d_batch, 'edge_index') and r3d_batch.edge_index.numel() > 0:
                r3d_node_feat = self.rna_3d_gnn(
                    r3d_batch.x,
                    r3d_batch.edge_index,
                    r3d_batch.edge_attr,
                    r3d_batch.batch
                )
                r3d_atom = global_mean_pool(r3d_node_feat, r3d_batch.batch)
            r_list.append(r3d_atom)
            r_masks.append(data.has_3d.view(-1, 1))

        # RNA 3D - 残基级 (Residue-level / 多尺度聚合)
        if self.use_rna_3d and self.use_residue_3d:
            r3d_res = torch.zeros(B, self.embed_dim, device=self.device)
            # 假设离线加载了 data.rna_3d_res
            r3d_res_batch = Batch.from_data_list(data.rna_3d_res).to(self.device)
            if hasattr(r3d_res_batch, 'edge_index') and r3d_res_batch.edge_index.numel() > 0:
                res_x = r3d_res_batch.x
                # 消融实验：若禁用扭转角，仅截取前3列坐标
                if not self.use_torsion:
                    res_x = res_x[:, :3]
                
                res_node_feat = self.res_3d_gnn(res_x, r3d_res_batch.edge_index, r3d_res_batch.edge_attr, r3d_res_batch.batch)
                r3d_res = global_mean_pool(res_node_feat, r3d_res_batch.batch)
            r_list.append(r3d_res); r_masks.append(data.has_3d.view(-1, 1))

        # RNA 语义
        if self.use_semantic:
            r_sem_vec = data.rna_sem_vec.view(B, -1)
            r_sem = self.rna_sem_proj(r_sem_vec)
            r_list.append(r_sem)
            r_masks.append((r_sem_vec.abs().sum(1, keepdim=True) > 1e-6).float())

        # RNA 模态融合
        if self.wo_dynamic_weight:
            r_fused = torch.stack(r_list, dim=1).mean(dim=1)
        else:
            r_msk = torch.cat(r_masks, dim=1)
            rw = F.softmax(self.rna_dyn(torch.cat(r_list, dim=1)).masked_fill(r_msk == 0, self.neg_inf), dim=1)
            r_fused = torch.sum(torch.stack(r_list, dim=1) * rw.unsqueeze(2), dim=1)

        # 3. 模态交互 (Cross-Attention or Concat)
        if self.wo_cross_attn:
            xc = self.fusion_concat(torch.cat([drug_fused, r_fused], dim=-1))
            dx_sq, rx_sq = drug_fused, r_fused
        else:
            dx, _ = self.MHA_drug_from_rna(drug_fused.unsqueeze(1), ro, ro, padding_mask=rm)
            rx, _ = self.MHA_rna_from_drug(r_fused.unsqueeze(1), drug_fused.unsqueeze(1), drug_fused.unsqueeze(1))
            xc, _ = self.inter_modality_attn([dx.squeeze(1), rx.squeeze(1)])
            dx_sq, rx_sq = dx.squeeze(1), rx.squeeze(1)

        # 预测输出
        out = self.out(self.fc2(F.relu(self.fc1(xc))))

        # 4. 辅助对比损失 (语义对齐)
        cl = torch.tensor(0.0, device=self.device)
        if self.training and self.use_semantic:
            cl += self.cl_loss(drug_g, m_sem)
            cl += self.cl_loss(r1d, r_sem)
            if self.use_rna_2d: cl += self.cl_loss(r2d, r_sem)
            if self.use_rna_3d and self.use_atom_3d:
                cl += self.cl_loss(r3d_atom, r_sem)
            # 对残基级特征同样进行对齐增强
            if self.use_rna_3d and self.use_residue_3d:
                cl += self.cl_loss(r3d_res, r_sem)

        return {'out': out, 'contrastive_loss': cl, 'rna_feat': rx_sq, 'drug_feat': dx_sq}