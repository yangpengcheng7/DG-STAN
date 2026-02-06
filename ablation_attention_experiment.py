"""
消融实验: 联合注意力 vs 分离注意力对比
用于回复审稿人R1-6的意见

实验设计:
- Joint Attention: 并行计算空间和时间注意力，然后融合
- Separate Attention: 顺序执行空间注意力，再执行时间注意力（当前实现）

运行方式:
    python ablation_attention_experiment.py --dataset PeMS04 --mode joint
    python ablation_attention_experiment.py --dataset PeMS04 --mode separate
    python ablation_attention_experiment.py --dataset PeMS08 --mode joint
    python ablation_attention_experiment.py --dataset PeMS08 --mode separate
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Tuple
import time
import argparse
import os
import json

# ----------------------
# 配置
# ----------------------
class ModelConfig:
    def __init__(self):
        self.seq_len = 12         # 输入长度
        self.pred_len = 3         # 输出长度 (先用3步测试)
        self.hidden_dim = 64      # 隐藏维度
        self.num_heads = 4        # 注意力头数
        self.num_gcn_layers = 3   # 图卷积层数
        self.batch_size = 32
        self.base_lr = 5e-4
        self.weight_decay = 1e-4
        self.epochs = 100
        self.patience = 15
        self.sigma = 100.0
        self.train_ratio = 0.6
        self.val_ratio = 0.2
        self.dropout = 0.1


# ----------------------
# 核心模块
# ----------------------

class ResidualConvBlock(nn.Module):
    """残差卷积块"""
    def __init__(self, in_dim: int, out_dim: int, kernel_size: int, padding: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size, padding=padding),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Conv1d(out_dim, out_dim, kernel_size, padding=padding),
            nn.BatchNorm1d(out_dim)
        )
        self.shortcut = nn.Conv1d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.main(x) + self.shortcut(x))


class SeparateAttention(nn.Module):
    """
    分离注意力机制 - 顺序执行
    先计算空间注意力，再计算时间注意力
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.spatial_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.temporal_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, batch_size: int, num_nodes: int, window_size: int) -> torch.Tensor:
        """
        x: [B*W, N, H] 或需要reshape的输入
        顺序: Spatial -> Temporal
        """
        hidden_dim = x.size(-1)
        
        # Step 1: Spatial Attention (对每个时间步，在节点间计算注意力)
        # x: [B*W, N, H]
        residual = x
        x_attn, _ = self.spatial_attn(x, x, x)
        x = self.norm1(residual + self.dropout(x_attn))  # [B*W, N, H]
        
        # Step 2: Reshape for Temporal Attention
        # [B*W, N, H] -> [B, W, N, H] -> [B*N, W, H]
        x = x.view(batch_size, window_size, num_nodes, hidden_dim)
        x = x.permute(0, 2, 1, 3)  # [B, N, W, H]
        x = x.reshape(batch_size * num_nodes, window_size, hidden_dim)  # [B*N, W, H]
        
        # Step 3: Temporal Attention (对每个节点，在时间步间计算注意力)
        residual = x
        x_attn, _ = self.temporal_attn(x, x, x)
        x = self.norm2(residual + self.dropout(x_attn))  # [B*N, W, H]
        
        # Reshape back: [B*N, W, H] -> [B, N, W, H] -> [B, W, N, H] -> [B*W, N, H]
        x = x.view(batch_size, num_nodes, window_size, hidden_dim)
        x = x.permute(0, 2, 1, 3)  # [B, W, N, H]
        x = x.reshape(batch_size * window_size, num_nodes, hidden_dim)  # [B*W, N, H]
        
        return x


class JointAttention(nn.Module):
    """
    联合注意力机制 - 并行计算后融合
    同时计算空间和时间注意力，然后通过可学习权重融合
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 空间注意力
        self.spatial_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_s = nn.LayerNorm(hidden_dim)
        
        # 时间注意力
        self.temporal_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_t = nn.LayerNorm(hidden_dim)
        
        # 融合层 - 将两个注意力结果融合
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, batch_size: int, num_nodes: int, window_size: int) -> torch.Tensor:
        """
        x: [B*W, N, H]
        并行计算空间和时间注意力，然后融合
        """
        hidden_dim = x.size(-1)
        
        # ===== 并行分支1: Spatial Attention =====
        # x: [B*W, N, H] - 对每个时间步，在节点间计算注意力
        x_spatial, _ = self.spatial_attn(x, x, x)
        x_spatial = self.norm_s(x + self.dropout(x_spatial))  # [B*W, N, H]
        
        # ===== 并行分支2: Temporal Attention =====
        # 需要reshape: [B*W, N, H] -> [B*N, W, H]
        x_for_temp = x.view(batch_size, window_size, num_nodes, hidden_dim)
        x_for_temp = x_for_temp.permute(0, 2, 1, 3)  # [B, N, W, H]
        x_for_temp = x_for_temp.reshape(batch_size * num_nodes, window_size, hidden_dim)  # [B*N, W, H]
        
        x_temporal, _ = self.temporal_attn(x_for_temp, x_for_temp, x_for_temp)
        x_temporal = self.norm_t(x_for_temp + self.dropout(x_temporal))  # [B*N, W, H]
        
        # Reshape back: [B*N, W, H] -> [B*W, N, H]
        x_temporal = x_temporal.view(batch_size, num_nodes, window_size, hidden_dim)
        x_temporal = x_temporal.permute(0, 2, 1, 3)  # [B, W, N, H]
        x_temporal = x_temporal.reshape(batch_size * window_size, num_nodes, hidden_dim)  # [B*W, N, H]
        
        # ===== 融合两个分支 =====
        # 拼接: [B*W, N, 2H]
        x_concat = torch.cat([x_spatial, x_temporal], dim=-1)
        
        # 通过融合层: [B*W, N, H]
        x_fused = self.fusion(x_concat)
        
        # 残差连接
        x_out = self.norm_out(x + x_fused)
        
        return x_out


class MultiScaleTemporal(nn.Module):
    """多尺度时间特征提取"""
    def __init__(self, hidden_dim: int, window_size: int = 12):
        super().__init__()
        self.window_size = window_size
        self.hidden_dim = hidden_dim

        self.conv_blocks = nn.ModuleList([
            ResidualConvBlock(hidden_dim, hidden_dim, ks, ks//2)
            for ks in [3, 5, 11]
        ])

        self.fusion = nn.Sequential(
            nn.Conv1d(3 * hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # [B*N, H, W]
        features = []

        for conv in self.conv_blocks:
            conv_out = conv(x)
            features.append(conv_out)

        fused = torch.cat(features, dim=1)
        output = self.fusion(fused)

        return output.permute(0, 2, 1)


class GLU(nn.Module):
    """门控线性单元"""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) * torch.sigmoid(self.gate(x))


class GraphConvLayer(nn.Module):
    """单层图卷积"""
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.glu = GLU(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        batch_size, window_size, num_nodes, hidden_dim = x.size()
        x_conv = torch.einsum('bnm,bwmh->bwnh', adj, x)
        x_conv = torch.clamp(x_conv, min=-10, max=10)
        x_out = self.glu(x_conv)
        x_out = self.dropout(x_out)
        return self.norm(x + x_out)


class MultiLayerGraphConv(nn.Module):
    """多层图卷积网络"""
    def __init__(self, hidden_dim: int, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, dropout) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, adj)
        return x


class GraphFusionGate(nn.Module):
    """静态与动态图融合门"""
    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, static_adj: torch.Tensor, dynamic_adj: torch.Tensor) -> torch.Tensor:
        combined = torch.stack([static_adj.expand_as(dynamic_adj), dynamic_adj], dim=-1)
        gate = self.gate(combined).squeeze(-1)
        return gate * static_adj + (1 - gate) * dynamic_adj


class DynamicGraphGenerator(nn.Module):
    """动态图结构生成器"""
    def __init__(self, num_nodes: int, feat_dim: int, hidden_dim: int):
        super().__init__()
        self.node_emb = nn.Parameter(torch.randn(num_nodes, hidden_dim) * 0.01)
        self.fc = nn.Linear(feat_dim, hidden_dim)
        nn.init.xavier_uniform_(self.fc.weight, gain=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_agg = x.mean(dim=2)
        h = self.fc(x_agg) + self.node_emb.unsqueeze(0)
        h = torch.clamp(h, min=-10, max=10)
        similarity = torch.bmm(h, h.transpose(1, 2))
        similarity = torch.clamp(similarity, min=-20, max=20)
        return torch.sigmoid(similarity)


class OutputProjection(nn.Module):
    """输出投影层"""
    def __init__(self, hidden_dim: int, pred_len: int, dropout: float = 0.1):
        super().__init__()
        self.pred_len = pred_len
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*N, H]
        return self.proj(x).unsqueeze(-1)  # [B*N, pred_len, 1]


class DGSTAN_Ablation(nn.Module):
    """
    DG-STAN 消融实验版本
    支持两种注意力模式: 'joint' 或 'separate'
    """
    def __init__(self, num_nodes: int, feat_dim: int, adj_matrix: np.ndarray, 
                 config: ModelConfig, attention_mode: str = 'joint'):
        super().__init__()
        self.register_buffer('static_adj', torch.FloatTensor(adj_matrix))
        self.hidden_dim = config.hidden_dim
        self.num_nodes = num_nodes
        self.attention_mode = attention_mode
        
        print(f"[INFO] 注意力模式: {attention_mode}")

        # 图相关组件
        self.dynamic_graph = DynamicGraphGenerator(num_nodes, feat_dim, config.hidden_dim)
        self.graph_fusion = GraphFusionGate()

        # 空间投影
        self.spatial_proj = nn.Conv1d(feat_dim, config.hidden_dim, kernel_size=1)

        # 多层图卷积
        self.graph_conv = MultiLayerGraphConv(config.hidden_dim, config.num_gcn_layers, config.dropout)

        # ===== 核心区别: 注意力模块 =====
        if attention_mode == 'joint':
            self.st_attn = JointAttention(config.hidden_dim, config.num_heads, config.dropout)
        elif attention_mode == 'separate':
            self.st_attn = SeparateAttention(config.hidden_dim, config.num_heads, config.dropout)
        else:
            raise ValueError(f"Unknown attention mode: {attention_mode}")

        # 时间特征提取
        self.temporal_net = MultiScaleTemporal(config.hidden_dim, config.seq_len)

        # 输出投影
        self.output_proj = OutputProjection(config.hidden_dim, config.pred_len, config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, window_size, _ = x.size()

        # 1. 图结构融合
        dynamic_adj = self.dynamic_graph(x)
        adj = self.graph_fusion(self.static_adj, dynamic_adj)

        # 2. 空间投影
        x = x.view(-1, window_size, x.size(3))
        x = self.spatial_proj(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = x.view(batch_size, num_nodes, window_size, self.hidden_dim)

        # 3. 多层图卷积
        x = x.permute(0, 2, 1, 3)  # [B, W, N, H]
        x = self.graph_conv(x, adj)

        # 4. ===== 时空注意力 (核心区别) =====
        x_attn = x.reshape(batch_size * window_size, num_nodes, -1)  # [B*W, N, H]
        x_attn = self.st_attn(x_attn, batch_size, num_nodes, window_size)  # [B*W, N, H]
        x_attn = x_attn.view(batch_size, window_size, num_nodes, self.hidden_dim)

        # 5. 时间特征提取
        x_time = x_attn.permute(0, 2, 1, 3)  # [B, N, W, H]
        x_time = x_time.reshape(batch_size * num_nodes, window_size, -1)
        x_time = self.temporal_net(x_time)  # [B*N, W, H]

        # 6. 时间维度聚合
        x_pooled = x_time.mean(dim=1)  # [B*N, H]

        # 7. 输出投影
        output = self.output_proj(x_pooled)  # [B*N, pred_len, 1]

        # 8. 重塑输出
        output = output.view(batch_size, num_nodes, -1, 1)  # [B, N, pred_len, 1]

        return output


# ----------------------
# 数据处理
# ----------------------
class TrafficDataset(Dataset):
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.targets[idx]


class DataProcessor:
    @staticmethod
    def create_sequences(data: torch.Tensor, input_len: int, output_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        num_nodes, total_time, num_features = data.shape
        num_samples = total_time - input_len - output_len + 1

        inputs = []
        targets = []

        for i in range(num_samples):
            inputs.append(data[:, i:i+input_len, :])
            targets.append(data[:, i+input_len:i+input_len+output_len, :])

        inputs = torch.stack(inputs, dim=0)
        targets = torch.stack(targets, dim=0)

        return inputs, targets


# ----------------------
# 训练与评估
# ----------------------
class ModelTrainer:
    def __init__(self, model: nn.Module, config: ModelConfig, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.config = config

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.base_lr,
            weight_decay=config.weight_decay
        )

        self.loss_fn = nn.SmoothL1Loss(beta=1.0)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

        self.scaler = GradScaler()

    def train_epoch(self, train_loader: DataLoader, val_loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        valid_batches = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            with autocast():
                preds = self.model(inputs)
                loss = self.loss_fn(preds, targets)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            valid_batches += 1

        avg_train_loss = total_loss / max(valid_batches, 1)

        # 验证集评估
        val_loss = None
        if val_loader:
            self.model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    with autocast():
                        preds = self.model(inputs)
                        loss = self.loss_fn(preds, targets)
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        total_val_loss += loss.item()

            val_loss = total_val_loss / len(val_loader)
            self.scheduler.step(val_loss)

        return avg_train_loss, val_loss


class ModelEvaluator:
    @staticmethod
    def evaluate(model: nn.Module, loader: DataLoader, data_mean: float, data_std: float, device: torch.device):
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(device)
                with autocast():
                    preds = model(inputs)
                all_preds.append(preds.cpu())
                all_labels.append(targets.cpu())

        preds = torch.cat(all_preds, dim=0)
        labels = torch.cat(all_labels, dim=0)

        # 反归一化
        preds = preds * data_std + data_mean
        labels = labels * data_std + data_mean

        # 计算指标
        mae = F.l1_loss(preds, labels).item()
        rmse = torch.sqrt(F.mse_loss(preds, labels)).item()

        # MAPE - 过滤小值
        mask = labels > 1.0
        if mask.sum() > 0:
            mape = torch.mean(torch.abs((labels[mask] - preds[mask]) / labels[mask])).item() * 100
        else:
            mape = 0.0

        # R²
        ss_res = torch.sum((labels - preds) ** 2).item()
        ss_tot = torch.sum((labels - labels.mean()) ** 2).item()
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        return mae, rmse, mape, r2


# ----------------------
# 主程序
# ----------------------
def load_data(dataset_name: str, config: ModelConfig):
    """加载数据集"""
    if dataset_name == 'PeMS04':
        data_file = '/data_ssd/other_models/数据集/PeMS04/PeMS04.npz'
        adj_file = '/data_ssd/other_models/数据集/PeMS04/PeMS04.csv'
    elif dataset_name == 'PeMS08':
        data_file = '/data_ssd/other_models/数据集/PeMS08/PeMS08.npz'
        adj_file = '/data_ssd/other_models/数据集/PeMS08/PeMS08.csv'
    elif dataset_name == 'METR-LA':
        data_file = '/data_ssd/other_models/数据集/METR-LA/metr-la.h5'
        adj_file = '/data_ssd/other_models/数据集/METR-LA/adj_mx.pkl'
        # METR-LA需要特殊处理
        import pandas as pd
        import pickle
        df = pd.read_hdf(data_file)
        pems_data = torch.FloatTensor(df.values).T.unsqueeze(-1)  # [N, T, 1]
        with open(adj_file, 'rb') as f:
            _, _, adj_matrix = pickle.load(f, encoding='latin1')
        return pems_data, adj_matrix
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # 加载PeMS数据
    data = np.load(data_file)
    pems_data = torch.FloatTensor(data['data'][:, :, 0:1])  # [T, N, 1]
    pems_data = pems_data.permute(1, 0, 2)  # [N, T, 1]
    
    # 加载邻接矩阵
    edges = np.loadtxt(adj_file, delimiter=',', skiprows=1)
    num_nodes = pems_data.size(0)
    adj_matrix = np.zeros((num_nodes, num_nodes))
    
    for edge in edges:
        i, j, dist = int(edge[0]), int(edge[1]), edge[2]
        if dist > 0:
            adj_matrix[i, j] = 1.0 / dist
            adj_matrix[j, i] = 1.0 / dist
    
    return pems_data, adj_matrix


def run_experiment(dataset_name: str, attention_mode: str, pred_len: int = 3, 
                   seed: int = 42, gpu_id: int = 0):
    """运行单次实验"""
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # 设备
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f'训练设备：{device}')
    
    # 配置
    config = ModelConfig()
    config.pred_len = pred_len
    
    # 加载数据
    print(f"\n加载数据集: {dataset_name}")
    pems_data, adj_matrix = load_data(dataset_name, config)
    num_nodes = pems_data.size(0)
    print(f"节点数: {num_nodes}, 时间步: {pems_data.size(1)}")
    
    # 归一化
    train_size = int(config.train_ratio * pems_data.size(1))
    train_data = pems_data[:, :train_size, :]
    data_mean = train_data.mean().item()
    data_std = train_data.std().item()
    pems_data = (pems_data - data_mean) / data_std
    
    print(f"训练集均值: {data_mean:.2f}, 标准差: {data_std:.2f}")
    
    # 准备数据集
    input_seq, target_seq = DataProcessor.create_sequences(
        pems_data, config.seq_len, config.pred_len
    )
    
    # 划分数据集
    train_size = int(config.train_ratio * len(input_seq))
    val_size = int(config.val_ratio * len(input_seq))
    
    train_inputs = input_seq[:train_size]
    train_targets = target_seq[:train_size]
    val_inputs = input_seq[train_size:train_size+val_size]
    val_targets = target_seq[train_size:train_size+val_size]
    test_inputs = input_seq[train_size+val_size:]
    test_targets = target_seq[train_size+val_size:]
    
    train_dataset = TrafficDataset(train_inputs, train_targets)
    val_dataset = TrafficDataset(val_inputs, val_targets)
    test_dataset = TrafficDataset(test_inputs, test_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    print(f"数据划分: 训练{len(train_dataset)} | 验证{len(val_dataset)} | 测试{len(test_dataset)}")
    
    # 创建模型
    model = DGSTAN_Ablation(
        num_nodes=num_nodes,
        feat_dim=1,
        adj_matrix=adj_matrix,
        config=config,
        attention_mode=attention_mode
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")
    
    # 训练
    trainer = ModelTrainer(model, config, device)
    
    print(f"\n{'='*60}")
    print(f"开始训练 - {dataset_name} - {attention_mode} attention - {pred_len}步预测")
    print(f"{'='*60}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = f'ablation_{dataset_name}_{attention_mode}_{pred_len}step_seed{seed}.pth'
    
    start_time = time.time()
    
    for epoch in range(config.epochs):
        train_loss, val_loss = trainer.train_epoch(train_loader, val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{config.epochs}] | 训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f}")
        
        if patience_counter >= config.patience:
            print(f"\n✅ Early stopping at epoch {epoch+1}")
            break
    
    total_time = time.time() - start_time
    print(f"训练总时间: {total_time:.1f}秒")
    
    # 加载最佳模型并测试
    model.load_state_dict(torch.load(best_model_path))
    
    test_mae, test_rmse, test_mape, test_r2 = ModelEvaluator.evaluate(
        model, test_loader, data_mean, data_std, device
    )
    
    print(f"\n{'='*60}")
    print(f"测试结果 - {dataset_name} - {attention_mode} - {pred_len}步")
    print(f"{'='*60}")
    print(f"MAE:  {test_mae:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"MAPE: {test_mape:.2f}%")
    print(f"R²:   {test_r2:.4f}")
    
    # 返回结果
    return {
        'dataset': dataset_name,
        'attention_mode': attention_mode,
        'pred_len': pred_len,
        'seed': seed,
        'MAE': test_mae,
        'RMSE': test_rmse,
        'MAPE': test_mape,
        'R2': test_r2,
        'train_time': total_time,
        'num_params': num_params
    }


def main():
    parser = argparse.ArgumentParser(description='消融实验: 联合注意力 vs 分离注意力')
    parser.add_argument('--dataset', type=str, default='PeMS04', 
                        choices=['PeMS04', 'PeMS08', 'METR-LA'])
    parser.add_argument('--mode', type=str, default='joint', 
                        choices=['joint', 'separate'])
    parser.add_argument('--pred_len', type=int, default=3, 
                        choices=[3, 6, 12])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seeds', type=str, default='42,123,456,789,1024',
                        help='逗号分隔的随机种子列表')
    parser.add_argument('--run_all', action='store_true',
                        help='运行所有配置的完整实验')
    
    args = parser.parse_args()
    
    if args.run_all:
        # 运行完整实验
        datasets = ['PeMS04', 'PeMS08']
        modes = ['joint', 'separate']
        pred_lens = [3]  # 先用3步测试
        seeds = [42, 123, 456]  # 3次重复
        
        all_results = []
        
        for dataset in datasets:
            for mode in modes:
                for pred_len in pred_lens:
                    for seed in seeds:
                        print(f"\n{'#'*70}")
                        print(f"# 实验: {dataset} | {mode} | {pred_len}步 | seed={seed}")
                        print(f"{'#'*70}")
                        
                        result = run_experiment(dataset, mode, pred_len, seed, args.gpu)
                        all_results.append(result)
                        
                        # 保存中间结果
                        with open('ablation_attention_results.json', 'w') as f:
                            json.dump(all_results, f, indent=2)
        
        # 汇总结果
        print("\n" + "="*80)
        print("实验结果汇总")
        print("="*80)
        
        # 按配置分组计算均值和标准差
        from collections import defaultdict
        grouped = defaultdict(list)
        for r in all_results:
            key = (r['dataset'], r['attention_mode'], r['pred_len'])
            grouped[key].append(r)
        
        print(f"{'Dataset':<10} {'Mode':<10} {'Steps':<6} {'MAE':<15} {'RMSE':<15} {'R²':<15}")
        print("-"*80)
        
        for key, results in sorted(grouped.items()):
            dataset, mode, pred_len = key
            maes = [r['MAE'] for r in results]
            rmses = [r['RMSE'] for r in results]
            r2s = [r['R2'] for r in results]
            
            mae_mean, mae_std = np.mean(maes), np.std(maes)
            rmse_mean, rmse_std = np.mean(rmses), np.std(rmses)
            r2_mean, r2_std = np.mean(r2s), np.std(r2s)
            
            print(f"{dataset:<10} {mode:<10} {pred_len:<6} "
                  f"{mae_mean:.2f}±{mae_std:.2f}   "
                  f"{rmse_mean:.2f}±{rmse_std:.2f}   "
                  f"{r2_mean:.4f}±{r2_std:.4f}")
    
    else:
        # 单次实验
        seeds = [int(s) for s in args.seeds.split(',')]
        
        results = []
        for seed in seeds:
            result = run_experiment(args.dataset, args.mode, args.pred_len, seed, args.gpu)
            results.append(result)
        
        # 计算均值
        if len(results) > 1:
            maes = [r['MAE'] for r in results]
            rmses = [r['RMSE'] for r in results]
            r2s = [r['R2'] for r in results]
            
            print(f"\n{'='*60}")
            print(f"多次实验汇总 ({len(results)}次)")
            print(f"{'='*60}")
            print(f"MAE:  {np.mean(maes):.4f} ± {np.std(maes):.4f}")
            print(f"RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
            print(f"R²:   {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")


if __name__ == '__main__':
    main()
