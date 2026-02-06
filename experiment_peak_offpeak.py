"""
实验2: Peak vs Off-peak 分析
来源: R1-8, R1-11, R4-1

目的: 验证动态图机制在不同交通条件下的表现

实验设计:
1. 将测试集按时段划分：
   - 非高峰 (0:00-6:00)
   - 早高峰 (7:00-9:00)
   - 中午 (10:00-16:00)
   - 晚高峰 (17:00-19:00)
   - 夜间 (20:00-24:00)

2. 分别计算各时段的MAE

3. 对比 DG-STAN vs 仅静态图版本

运行方式:
    python experiment_peak_offpeak.py --dataset PeMS04 --gpu 0
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Tuple, Dict, List
import time
import argparse
import os
from datetime import datetime, timedelta

# ----------------------
# 配置
# ----------------------
class ModelConfig:
    def __init__(self, pred_len=3):
        self.seq_len = 12
        self.pred_len = pred_len
        self.hidden_dim = 64
        self.num_heads = 4
        self.num_gcn_layers = 3
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
# 模型组件
# ----------------------

class ResidualConvBlock(nn.Module):
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


class SpatioTemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.spatial_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.temporal_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_attn, _ = self.spatial_attn(x, x, x)
        x = self.norm1(residual + self.dropout(x_attn))
        residual = x
        x_attn, _ = self.temporal_attn(x, x, x)
        x = self.norm2(residual + self.dropout(x_attn))
        return x


class MultiScaleTemporal(nn.Module):
    def __init__(self, hidden_dim: int, window_size: int = 12):
        super().__init__()
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
        x = x.permute(0, 2, 1)
        features = [conv(x) for conv in self.conv_blocks]
        fused = torch.cat(features, dim=1)
        output = self.fusion(fused)
        return output.permute(0, 2, 1)


class GLU(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) * torch.sigmoid(self.gate(x))


class GraphConvLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.glu = GLU(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x_conv = torch.einsum('bnm,bwmh->bwnh', adj, x)
        x_conv = torch.clamp(x_conv, min=-10, max=10)
        x_out = self.glu(x_conv)
        x_out = self.dropout(x_out)
        return self.norm(x + x_out)


class MultiLayerGraphConv(nn.Module):
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
    def __init__(self, hidden_dim: int, pred_len: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).unsqueeze(-1)


class DGSTAN(nn.Module):
    """
    DG-STAN 模型
    use_dynamic_graph: True = 完整模型, False = 仅静态图版本
    """
    def __init__(self, num_nodes: int, feat_dim: int, adj_matrix: np.ndarray, 
                 config: ModelConfig, use_dynamic_graph: bool = True):
        super().__init__()
        self.register_buffer('static_adj', torch.FloatTensor(adj_matrix))
        self.hidden_dim = config.hidden_dim
        self.num_nodes = num_nodes
        self.use_dynamic_graph = use_dynamic_graph
        
        if use_dynamic_graph:
            self.dynamic_graph = DynamicGraphGenerator(num_nodes, feat_dim, config.hidden_dim)
            self.graph_fusion = GraphFusionGate()

        self.spatial_proj = nn.Conv1d(feat_dim, config.hidden_dim, kernel_size=1)
        self.graph_conv = MultiLayerGraphConv(config.hidden_dim, config.num_gcn_layers, config.dropout)
        self.st_attn = SpatioTemporalAttention(config.hidden_dim, config.num_heads, config.dropout)
        self.temporal_net = MultiScaleTemporal(config.hidden_dim, config.seq_len)
        self.output_proj = OutputProjection(config.hidden_dim, config.pred_len, config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, window_size, _ = x.size()

        # 图结构
        if self.use_dynamic_graph:
            dynamic_adj = self.dynamic_graph(x)
            adj = self.graph_fusion(self.static_adj, dynamic_adj)
        else:
            # 仅使用静态图
            adj = self.static_adj.unsqueeze(0).expand(batch_size, -1, -1)

        # 空间投影
        x = x.view(-1, window_size, x.size(3))
        x = self.spatial_proj(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = x.view(batch_size, num_nodes, window_size, self.hidden_dim)

        # 图卷积
        x = x.permute(0, 2, 1, 3)
        x = self.graph_conv(x, adj)

        # 时空注意力
        x_attn = x.reshape(batch_size * window_size, num_nodes, -1)
        x_attn = self.st_attn(x_attn)
        x_attn = x_attn.view(batch_size, window_size, num_nodes, self.hidden_dim)

        # 时间特征
        x_time = x_attn.permute(0, 2, 1, 3)
        x_time = x_time.reshape(batch_size * num_nodes, window_size, -1)
        x_time = self.temporal_net(x_time)

        # 聚合和输出
        x_pooled = x_time.mean(dim=1)
        output = self.output_proj(x_pooled)
        output = output.view(batch_size, num_nodes, -1, 1)

        return output


# ----------------------
# 数据处理
# ----------------------
class TrafficDataset(Dataset):
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor, timestamps: np.ndarray = None):
        self.inputs = inputs
        self.targets = targets
        self.timestamps = timestamps

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx: int):
        if self.timestamps is not None:
            return self.inputs[idx], self.targets[idx], self.timestamps[idx]
        return self.inputs[idx], self.targets[idx]


def create_sequences_with_timestamps(data: torch.Tensor, timestamps: np.ndarray, 
                                     input_len: int, output_len: int):
    """创建序列并保留时间戳"""
    num_nodes, total_time, num_features = data.shape
    num_samples = total_time - input_len - output_len + 1

    inputs = []
    targets = []
    target_timestamps = []

    for i in range(num_samples):
        inputs.append(data[:, i:i+input_len, :])
        targets.append(data[:, i+input_len:i+input_len+output_len, :])
        # 保存预测目标的起始时间戳
        target_timestamps.append(timestamps[i+input_len])

    inputs = torch.stack(inputs, dim=0)
    targets = torch.stack(targets, dim=0)
    target_timestamps = np.array(target_timestamps)

    return inputs, targets, target_timestamps


def get_time_period(hour: int) -> str:
    """根据小时判断时段"""
    if 0 <= hour < 6:
        return 'off_peak_night'  # 非高峰 (0:00-6:00)
    elif 6 <= hour < 10:
        return 'morning_peak'    # 早高峰 (6:00-10:00)
    elif 10 <= hour < 16:
        return 'midday'          # 中午 (10:00-16:00)
    elif 16 <= hour < 20:
        return 'evening_peak'    # 晚高峰 (16:00-20:00)
    else:
        return 'night'           # 夜间 (20:00-24:00)


# ----------------------
# 训练与评估
# ----------------------
class ModelTrainer:
    def __init__(self, model: nn.Module, config: ModelConfig, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.base_lr, weight_decay=config.weight_decay)
        self.loss_fn = nn.SmoothL1Loss(beta=1.0)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)
        self.scaler = GradScaler()

    def train_epoch(self, train_loader: DataLoader, val_loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        valid_batches = 0

        for batch in train_loader:
            inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
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

        # 验证
        val_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                with autocast():
                    preds = self.model(inputs)
                    loss = self.loss_fn(preds, targets)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        self.scheduler.step(val_loss)

        return avg_train_loss, val_loss


def evaluate_by_time_period(model: nn.Module, test_loader: DataLoader, 
                            data_mean: float, data_std: float, device: torch.device) -> Dict:
    """按时段评估模型"""
    model.eval()
    
    # 存储每个时段的预测和真实值
    period_results = {
        'off_peak_night': {'preds': [], 'labels': []},
        'morning_peak': {'preds': [], 'labels': []},
        'midday': {'preds': [], 'labels': []},
        'evening_peak': {'preds': [], 'labels': []},
        'night': {'preds': [], 'labels': []},
    }
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs, targets, timestamps = batch[0].to(device), batch[1], batch[2]
            
            with autocast():
                preds = model(inputs)
            
            preds = preds.cpu()
            
            # 反归一化
            preds_denorm = preds * data_std + data_mean
            targets_denorm = targets * data_std + data_mean
            
            # 按时段分类
            for i in range(len(timestamps)):
                ts = timestamps[i]
                # 将时间戳转换为小时 (每5分钟一个样本, 288个样本/天)
                # ts是数据的全局索引
                time_of_day = ts % 288  # 一天288个5分钟间隔
                hour = (time_of_day * 5) // 60  # 转换为小时
                
                period = get_time_period(hour)
                period_results[period]['preds'].append(preds_denorm[i].unsqueeze(0))
                period_results[period]['labels'].append(targets_denorm[i].unsqueeze(0))
            
            all_preds.append(preds_denorm)
            all_labels.append(targets_denorm)
    
    # 计算每个时段的MAE
    results = {}
    for period, data in period_results.items():
        if len(data['preds']) > 0:
            preds = torch.cat(data['preds'], dim=0)
            labels = torch.cat(data['labels'], dim=0)
            
            mae = F.l1_loss(preds, labels).item()
            results[period] = {
                'MAE': mae,
                'samples': len(data['preds'])
            }
        else:
            results[period] = {'MAE': 0, 'samples': 0}
    
    # 计算总体MAE
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    overall_mae = F.l1_loss(all_preds, all_labels).item()
    results['overall'] = {'MAE': overall_mae, 'samples': len(all_preds)}
    
    return results


def load_data(dataset_name: str):
    """加载数据集"""
    if dataset_name == 'PeMS04':
        data_file = '/data_ssd/other_models/数据集/PeMS04/PeMS04.npz'
        adj_file = '/data_ssd/other_models/数据集/PeMS04/PeMS04.csv'
    elif dataset_name == 'PeMS08':
        data_file = '/data_ssd/other_models/数据集/PeMS08/PeMS08.npz'
        adj_file = '/data_ssd/other_models/数据集/PeMS08/PeMS08.csv'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    data = np.load(data_file)
    pems_data = torch.FloatTensor(data['data'][:, :, 0:1])
    pems_data = pems_data.permute(1, 0, 2)
    
    # 创建时间戳 (假设数据从0点开始，每5分钟一个样本)
    total_time = pems_data.size(1)
    timestamps = np.arange(total_time)  # 简单使用索引作为时间戳
    
    edges = np.loadtxt(adj_file, delimiter=',', skiprows=1)
    num_nodes = pems_data.size(0)
    adj_matrix = np.zeros((num_nodes, num_nodes))
    
    for edge in edges:
        i, j, dist = int(edge[0]), int(edge[1]), edge[2]
        if dist > 0:
            adj_matrix[i, j] = 1.0 / dist
            adj_matrix[j, i] = 1.0 / dist
    
    return pems_data, adj_matrix, timestamps


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                config: ModelConfig, device: torch.device, model_name: str) -> nn.Module:
    """训练模型"""
    trainer = ModelTrainer(model, config, device)
    
    best_val_loss = float('inf')
    patience_counter = 0
    model_path = f'{model_name}_best.pth'
    
    print(f"\n开始训练 {model_name}...")
    
    for epoch in range(config.epochs):
        train_loss, val_loss = trainer.train_epoch(train_loader, val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{config.epochs}] | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        
        if patience_counter >= config.patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(torch.load(model_path))
    return model


def main():
    parser = argparse.ArgumentParser(description='Peak vs Off-peak 分析实验')
    parser.add_argument('--dataset', type=str, default='PeMS04', choices=['PeMS04', 'PeMS08'])
    parser.add_argument('--pred_len', type=int, default=3, choices=[3, 6, 12])
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("实验2: Peak vs Off-peak 分析")
    print(f"数据集: {args.dataset} | 预测步数: {args.pred_len}")
    print("=" * 70)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')
    
    config = ModelConfig(pred_len=args.pred_len)
    
    # 加载数据
    print("\n加载数据...")
    pems_data, adj_matrix, timestamps = load_data(args.dataset)
    num_nodes = pems_data.size(0)
    print(f"节点数: {num_nodes}, 时间步: {pems_data.size(1)}")
    
    # 归一化
    train_size = int(config.train_ratio * pems_data.size(1))
    train_data = pems_data[:, :train_size, :]
    data_mean = train_data.mean().item()
    data_std = train_data.std().item()
    pems_data_norm = (pems_data - data_mean) / data_std
    
    # 创建序列
    input_seq, target_seq, target_timestamps = create_sequences_with_timestamps(
        pems_data_norm, timestamps, config.seq_len, config.pred_len
    )
    
    # 划分数据集
    train_size = int(config.train_ratio * len(input_seq))
    val_size = int(config.val_ratio * len(input_seq))
    
    train_inputs = input_seq[:train_size]
    train_targets = target_seq[:train_size]
    train_ts = target_timestamps[:train_size]
    
    val_inputs = input_seq[train_size:train_size+val_size]
    val_targets = target_seq[train_size:train_size+val_size]
    val_ts = target_timestamps[train_size:train_size+val_size]
    
    test_inputs = input_seq[train_size+val_size:]
    test_targets = target_seq[train_size+val_size:]
    test_ts = target_timestamps[train_size+val_size:]
    
    train_dataset = TrafficDataset(train_inputs, train_targets, train_ts)
    val_dataset = TrafficDataset(val_inputs, val_targets, val_ts)
    test_dataset = TrafficDataset(test_inputs, test_targets, test_ts)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    print(f"数据划分: 训练{len(train_dataset)} | 验证{len(val_dataset)} | 测试{len(test_dataset)}")
    
    # ===== 训练完整DG-STAN模型 =====
    print("\n" + "=" * 50)
    print("训练 DG-STAN (动态图 + 静态图)")
    print("=" * 50)
    
    model_dgstan = DGSTAN(
        num_nodes=num_nodes,
        feat_dim=1,
        adj_matrix=adj_matrix,
        config=config,
        use_dynamic_graph=True
    )
    print(f"参数量: {sum(p.numel() for p in model_dgstan.parameters()):,}")
    
    model_dgstan = train_model(
        model_dgstan, train_loader, val_loader, config, device,
        f'dgstan_full_{args.dataset}_{args.pred_len}step'
    )
    
    # ===== 训练仅静态图版本 =====
    print("\n" + "=" * 50)
    print("训练 Static-only (仅静态图)")
    print("=" * 50)
    
    model_static = DGSTAN(
        num_nodes=num_nodes,
        feat_dim=1,
        adj_matrix=adj_matrix,
        config=config,
        use_dynamic_graph=False
    )
    print(f"参数量: {sum(p.numel() for p in model_static.parameters()):,}")
    
    model_static = train_model(
        model_static, train_loader, val_loader, config, device,
        f'dgstan_static_{args.dataset}_{args.pred_len}step'
    )
    
    # ===== 按时段评估 =====
    print("\n" + "=" * 70)
    print("按时段评估结果")
    print("=" * 70)
    
    results_dgstan = evaluate_by_time_period(model_dgstan, test_loader, data_mean, data_std, device)
    results_static = evaluate_by_time_period(model_static, test_loader, data_mean, data_std, device)
    
    # 打印结果表格
    period_names = {
        'off_peak_night': '非高峰 (0-6h)',
        'morning_peak': '早高峰 (7-10h)',
        'midday': '中午 (10-16h)',
        'evening_peak': '晚高峰 (17-20h)',
        'night': '夜间 (20-24h)',
        'overall': '总体'
    }
    
    print(f"\n{'时段':<20} {'DG-STAN MAE':<15} {'Static MAE':<15} {'改进%':<10} {'样本数':<10}")
    print("-" * 70)
    
    for period in ['off_peak_night', 'morning_peak', 'midday', 'evening_peak', 'night', 'overall']:
        dgstan_mae = results_dgstan[period]['MAE']
        static_mae = results_static[period]['MAE']
        samples = results_dgstan[period]['samples']
        
        if static_mae > 0:
            improvement = (static_mae - dgstan_mae) / static_mae * 100
        else:
            improvement = 0
        
        print(f"{period_names[period]:<20} {dgstan_mae:<15.4f} {static_mae:<15.4f} {improvement:<10.2f}% {samples:<10}")
    
    # 保存结果
    result_file = f'peak_offpeak_results_{args.dataset}_{args.pred_len}step.txt'
    with open(result_file, 'w') as f:
        f.write(f"Peak vs Off-peak Analysis Results\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Prediction Length: {args.pred_len} steps\n")
        f.write(f"\n{'时段':<20} {'DG-STAN MAE':<15} {'Static MAE':<15} {'改进%':<10}\n")
        f.write("-" * 60 + "\n")
        
        for period in ['off_peak_night', 'morning_peak', 'midday', 'evening_peak', 'night', 'overall']:
            dgstan_mae = results_dgstan[period]['MAE']
            static_mae = results_static[period]['MAE']
            if static_mae > 0:
                improvement = (static_mae - dgstan_mae) / static_mae * 100
            else:
                improvement = 0
            f.write(f"{period_names[period]:<20} {dgstan_mae:<15.4f} {static_mae:<15.4f} {improvement:<10.2f}%\n")
    
    print(f"\n结果已保存到: {result_file}")
    print("=" * 70)


if __name__ == '__main__':
    main()
