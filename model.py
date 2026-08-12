import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import AttentionalAggregation, GINConv, MessagePassing, RGCNConv, global_mean_pool


# =========================================================
# Basic modules
# =========================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class MyMHA(nn.Module):
    def __init__(self, args):
        super().__init__()

        embed_dim = args["embed_dim"]
        num_heads = args["nhead"]

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by nhead ({num_heads}).")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.prior_weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, x_q, x_k, x_v, prior_attn=None, padding_mask=None):
        batch_size, query_len, embed_dim = x_q.shape
        key_len = x_k.shape[1]
        num_heads, head_dim = self.num_heads, self.head_dim

        Q = self.q_proj(x_q).view(batch_size, query_len, num_heads, head_dim).transpose(1, 2)
        K = self.k_proj(x_k).view(batch_size, key_len, num_heads, head_dim).transpose(1, 2)
        V = self.v_proj(x_v).view(batch_size, key_len, num_heads, head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)

        if padding_mask is not None:
            attn_scores = attn_scores.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)

        if prior_attn is not None:
            weight = torch.sigmoid(self.prior_weight)
            attn_weights = (1 - weight) * attn_weights + weight * prior_attn.unsqueeze(1)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, query_len, embed_dim)

        return self.out_proj(attn_output), attn_weights


def masked_mean_pooling(x, mask):
    valid = (~mask[:, :x.size(1)]).unsqueeze(-1).float()
    return (x * valid).sum(1) / valid.sum(1).clamp(min=1)


# =========================================================
# Modality encoders
# =========================================================

class MultiScaleGIN(nn.Module):
    def __init__(self, dim_atom, dim_hid, dropout):
        super().__init__()

        def make_mlp(in_dim, out_dim):
            return nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))

        self.conv1 = GINConv(make_mlp(dim_atom, dim_hid))
        self.conv2 = GINConv(make_mlp(dim_hid, dim_hid))
        self.conv3 = GINConv(make_mlp(dim_hid, dim_hid))

        self.norm1 = nn.LayerNorm(dim_hid)
        self.norm2 = nn.LayerNorm(dim_hid)
        self.norm3 = nn.LayerNorm(dim_hid)

        self.pool = AttentionalAggregation(nn.Sequential(nn.Linear(dim_hid, 1), nn.ReLU()))
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch):
        x1 = self.relu(self.norm1(self.conv1(x, edge_index)))
        x2 = self.relu(self.norm2(self.conv2(x1, edge_index)))
        x3 = self.relu(self.norm3(self.conv3(x2, edge_index)))

        graph_feature = torch.cat(
            [self.pool(x1, batch), self.pool(x2, batch), self.pool(x3, batch)],
            dim=1,
        )
        node_feature = torch.cat([x1, x2, x3], dim=1)
        return graph_feature, node_feature


class RNA_RGCN(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.conv1 = RGCNConv(args["num_features_rna"], args["embed_dim"], args["num_relations"])
        self.conv2 = RGCNConv(args["embed_dim"], args["embed_dim"], args["num_relations"])
        self.fc = nn.Linear(args["embed_dim"], args["embed_dim"])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args["dropout"])

    def forward(self, data):
        x = self.relu(self.conv1(data.x, data.edge_index, data.edge_type))
        x = self.conv2(x, data.edge_index, data.edge_type)
        x = global_mean_pool(x, data.batch)
        return self.dropout(self.relu(self.fc(x)))


class GeometricGNN(MessagePassing):
    def __init__(self, hidden_dim, node_dim=9, edge_dim=1):
        super().__init__(aggr="add")
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

    def forward(self, x, edge_index, edge_attr, batch):
        return self.output_mlp(self.propagate(edge_index, x=x, edge_attr=edge_attr, batch=batch))

    def message(self, x_j, edge_attr):
        return self.message_mlp(torch.cat([x_j, edge_attr], dim=-1))


# =========================================================
# Losses and fusion modules
# =========================================================

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, x1, x2):
        valid = (x1.abs().sum(1) > 1e-6) & (x2.abs().sum(1) > 1e-6)

        if valid.sum() < 2:
            return torch.tensor(0.0, requires_grad=True, device=x1.device)

        similarity = F.cosine_similarity(
            x1[valid].unsqueeze(1),
            x2[valid].unsqueeze(0),
            dim=-1,
        ) / self.temperature
        target = torch.arange(similarity.size(0), device=similarity.device)

        return F.cross_entropy(similarity, target)


class InteractionContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, rna_feature, drug_feature, label):
        similarity = F.cosine_similarity(
            F.normalize(rna_feature, dim=1),
            F.normalize(drug_feature, dim=1),
        ) / self.temperature

        positive = label == 1
        negative = label == 0

        positive_loss = (1 - similarity[positive]).mean() if positive.sum() > 0 else similarity.new_tensor(0.0)
        negative_loss = (similarity[negative] + 1).clamp(min=0).mean() if negative.sum() > 0 else similarity.new_tensor(0.0)

        return positive_loss + negative_loss


class DynamicWeightNet(nn.Module):
    def __init__(self, input_dim, num_modalities):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_modalities)

    def forward(self, x):
        return self.fc(x)


class InterModalityAttention(nn.Module):
    def __init__(self, embed_dim, num_branches=2):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 1)
        self.num_branches = num_branches

    def forward(self, features):
        stacked = torch.stack(features, dim=1)
        weights = F.softmax(self.fc(stacked).squeeze(-1), dim=-1)
        fused = torch.sum(stacked * weights.unsqueeze(2), dim=1)
        return fused, weights


# =========================================================
# GeoM3-RMI
# =========================================================

class MultiModalModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.device = args["device"]
        self.embed_dim = args["embed_dim"]
        self.neg_inf = -1e9

        # Small-molecule graph branch
        self.drug_gin = MultiScaleGIN(args["dim_atom"], args["embed_dim"], args["dropout"])
        self.drug_proj = nn.Linear(3 * args["embed_dim"], args["embed_dim"])

        # RNA 1D branch
        self.embedding_xt = nn.Embedding(
            args["num_base"] ** args["k_value"] + 1,
            args["embed_dim"],
            padding_idx=0,
        )
        self.pos_encoder = PositionalEncoding(args["embed_dim"], args["max_seq_len"])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args["embed_dim"],
            nhead=args["nhead"],
            batch_first=True,
            dropout=args["dropout"],
        )
        self.rna_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=args["transformer_encoder_layer"],
        )
        self.rna_fc = nn.Linear(args["embed_dim"], args["embed_dim"])

        # RNA 2D branch
        self.rna_rgcn = RNA_RGCN(args)

        # RNA dual-scale 3D branches
        self.rna_3d_gnn = GeometricGNN(args["embed_dim"], args["node_dim_3d"], args["edge_dim_3d"])
        self.res_3d_gnn = GeometricGNN(args["embed_dim"], 9, 1)  # xyz + six torsion angles

        # Semantic projections
        self.mol_sem_proj = nn.Linear(768, args["embed_dim"])
        self.rna_sem_proj = nn.Linear(768, args["embed_dim"])

        # Dynamic modality weighting
        self.drug_dyn = DynamicWeightNet(2 * self.embed_dim, 2)
        self.rna_dyn = DynamicWeightNet(5 * self.embed_dim, 5)

        # Dual asymmetric cross-attention
        self.MHA_drug_from_rna = MyMHA(args)
        self.MHA_rna_from_drug = MyMHA(args)
        self.inter_modality_attn = InterModalityAttention(self.embed_dim, 2)

        # Prediction head
        self.fc1 = nn.Linear(self.embed_dim, args["embed_dim"] * 2)
        self.fc2 = nn.Linear(args["embed_dim"] * 2, args["embed_dim"])
        self.out = nn.Linear(args["embed_dim"], args["n_output"])

        self.cl_loss = ContrastiveLoss(temperature=args.get("contrastive_temp", 0.1))

    def forward(self, data):
        batch_size = data.y.size(0)

        # -----------------------------------------------------
        # 1. Small-molecule representations
        # -----------------------------------------------------
        drug_batch = Batch.from_data_list(data.drug).to(self.device)
        drug_multiscale, _ = self.drug_gin(drug_batch.x, drug_batch.edge_index, drug_batch.batch)
        drug_graph = self.drug_proj(drug_multiscale)

        mol_sem_vec = data.mol_sem_vec.view(batch_size, -1)
        mol_semantic = self.mol_sem_proj(mol_sem_vec)

        mol_features = [drug_graph, mol_semantic]
        mol_masks = [
            torch.ones(batch_size, 1, device=self.device),
            (mol_sem_vec.abs().sum(1, keepdim=True) > 1e-6).float(),
        ]

        mol_mask = torch.cat(mol_masks, dim=1)
        mol_logits = self.drug_dyn(torch.cat(mol_features, dim=1)).masked_fill(mol_mask == 0, self.neg_inf)
        mol_weights = F.softmax(mol_logits, dim=1)
        drug_fused = torch.sum(torch.stack(mol_features, dim=1) * mol_weights.unsqueeze(2), dim=1)

        # -----------------------------------------------------
        # 2. RNA 1D representation
        # -----------------------------------------------------
        rna_sequence = data.rna_sequence.view(batch_size, -1).long()
        padding_mask = rna_sequence == 0

        rna_embedding = self.pos_encoder(self.embedding_xt(rna_sequence))
        H_1D = self.rna_transformer(rna_embedding, src_key_padding_mask=padding_mask)
        rna_1d = masked_mean_pooling(self.rna_fc(H_1D), padding_mask)

        # -----------------------------------------------------
        # 3. RNA 2D representation
        # -----------------------------------------------------
        rna_2d_batch = Batch.from_data_list(data.rna_2d).to(self.device)
        rna_2d = self.rna_rgcn(rna_2d_batch)

        # -----------------------------------------------------
        # 4. RNA atom-level 3D representation
        # -----------------------------------------------------
        rna_3d_atom = torch.zeros(batch_size, self.embed_dim, device=self.device)
        atom_batch = Batch.from_data_list(data.rna_3d).to(self.device)

        if hasattr(atom_batch, "edge_index") and atom_batch.edge_index.numel() > 0:
            atom_nodes = self.rna_3d_gnn(
                atom_batch.x,
                atom_batch.edge_index,
                atom_batch.edge_attr,
                atom_batch.batch,
            )
            rna_3d_atom = global_mean_pool(atom_nodes, atom_batch.batch)

        # -----------------------------------------------------
        # 5. RNA residue-level 3D representation
        # -----------------------------------------------------
        rna_3d_residue = torch.zeros(batch_size, self.embed_dim, device=self.device)
        residue_batch = Batch.from_data_list(data.rna_3d_res).to(self.device)

        if hasattr(residue_batch, "edge_index") and residue_batch.edge_index.numel() > 0:
            residue_nodes = self.res_3d_gnn(
                residue_batch.x,
                residue_batch.edge_index,
                residue_batch.edge_attr,
                residue_batch.batch,
            )
            rna_3d_residue = global_mean_pool(residue_nodes, residue_batch.batch)

        # -----------------------------------------------------
        # 6. RNA semantic representation and dynamic fusion
        # -----------------------------------------------------
        rna_sem_vec = data.rna_sem_vec.view(batch_size, -1)
        rna_semantic = self.rna_sem_proj(rna_sem_vec)

        rna_features = [rna_1d, rna_2d, rna_3d_atom, rna_3d_residue, rna_semantic]
        rna_masks = [
            torch.ones(batch_size, 1, device=self.device),
            torch.ones(batch_size, 1, device=self.device),
            data.has_3d.view(-1, 1),
            data.has_3d.view(-1, 1),
            (rna_sem_vec.abs().sum(1, keepdim=True) > 1e-6).float(),
        ]

        rna_mask = torch.cat(rna_masks, dim=1)
        rna_logits = self.rna_dyn(torch.cat(rna_features, dim=1)).masked_fill(rna_mask == 0, self.neg_inf)
        rna_weights = F.softmax(rna_logits, dim=1)
        rna_fused = torch.sum(torch.stack(rna_features, dim=1) * rna_weights.unsqueeze(2), dim=1)

        # -----------------------------------------------------
        # 7. Asymmetric dual-branch cross-entity interaction
        # -----------------------------------------------------
        h_int, _ = self.MHA_drug_from_rna(
            drug_fused.unsqueeze(1),
            H_1D,
            H_1D,
            padding_mask=padding_mask,
        )
        h_cmp, _ = self.MHA_rna_from_drug(
            rna_fused.unsqueeze(1),
            drug_fused.unsqueeze(1),
            drug_fused.unsqueeze(1),
        )

        h_int = h_int.squeeze(1)
        h_cmp = h_cmp.squeeze(1)
        F_fused, _ = self.inter_modality_attn([h_int, h_cmp])

        # -----------------------------------------------------
        # 8. Prediction
        # -----------------------------------------------------
        out = self.out(self.fc2(F.relu(self.fc1(F_fused))))

        # -----------------------------------------------------
        # 9. Modality-level contrastive alignment
        # -----------------------------------------------------
        contrastive_loss = torch.tensor(0.0, device=self.device)

        if self.training:
            contrastive_loss = (
                self.cl_loss(drug_graph, mol_semantic)
                + self.cl_loss(rna_1d, rna_semantic)
                + self.cl_loss(rna_2d, rna_semantic)
                + self.cl_loss(rna_3d_atom, rna_semantic)
                + self.cl_loss(rna_3d_residue, rna_semantic)
            )

        return {
            "out": out,
            "contrastive_loss": contrastive_loss,
            "rna_feat": h_cmp,
            "drug_feat": h_int,
        }
