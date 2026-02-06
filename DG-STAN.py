# -*- coding: utf-8 -*-
"""
主要修复：
1. 只预测flow特征（与基线一致）
2. 使用训练集统计量归一化
3. 添加数值稳定性保护
4. 降低初始学习率
5. 增强梯度裁剪
"""
from typing import Tuple, Dict, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------
# 配置管理
# ----------------------
class ModelConfig:
    """模型超参数配置"""
    def __init__(self):
        self.seq_len = 12         # 输入序列长度
        self.pred_len = 12        # 预测长度（修改为12步，与基线一致）
        self.hidden_dim = 64      # 隐藏层维度
        self.num_heads = 4        # 注意力头数
        self.batch_size = 32      # 批量大小
        self.base_lr = 5e-4       # 降低学习率：1e-3 → 5e-4
        self.weight_decay = 1e-4  # 权重衰减
        self.epochs = 100         # 增加训练轮数到100
        self.sigma = 100.0        # 邻接矩阵高斯核标准差
        self.train_ratio = 0.6    # 训练集比例（修改为60%，与基线一致）
        self.val_ratio = 0.2      # 验证集比例（修改为20%，与基线一致）

# ----------------------
# 数据处理模块
# ----------------------
class DataProcessor:
    @staticmethod
    def build_adjacency_matrix(csv_path: str) -> Tuple[np.ndarray, Dict[int, int]]:
        """构建带节点索引的邻接矩阵"""
        df = pd.read_csv(csv_path)
        nodes = sorted(set(df['from']).union(df['to']))
        node_map = {n: i for i, n in enumerate(nodes)}

        from_idx = df['from'].map(node_map).values.astype(int)
        to_idx = df['to'].map(node_map).values.astype(int)
        cost = df['cost'].values.astype(np.float32)

        adj_matrix = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
        adj_matrix[from_idx, to_idx] = cost
        adj_matrix[to_idx, from_idx] = cost

        print(f"邻接矩阵: {len(nodes)}个节点")
        return adj_matrix, node_map

    @staticmethod
    def normalize_adjacency(adj_matrix: np.ndarray, sigma: float) -> np.ndarray:
        """邻接矩阵归一化处理"""
        adj_normalized = np.exp(-(adj_matrix ** 2) / (2 * sigma ** 2))
        np.fill_diagonal(adj_normalized, 1)
        return adj_normalized

    @staticmethod
    def load_flow_with_train_stats(npz_path: str, train_ratio: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        修复1：只加载flow特征
        修复2：使用训练集统计量归一化
        """
        data = np.load(npz_path)['data'][:, :, 0:1]  # 只取flow

        # 计算训练集统计量
        total_time = data.shape[0]
        train_end = int(total_time * train_ratio)
        train_data = data[:train_end, :, :]

        mean = train_data.mean()
        std = train_data.std() + 1e-5

        # 归一化
        normalized_data = (data - mean) / std

        print(f"Flow数据: {data.shape}")
        print(f"训练集均值: {mean:.2f}, 标准差: {std:.2f}")

        return (
            torch.FloatTensor(normalized_data).permute(1, 0, 2).to(device),
            torch.FloatTensor([mean]).to(device),
            torch.FloatTensor([std]).to(device)
        )

    @staticmethod
    def create_sequences(data: torch.Tensor, input_len: int, output_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        时间序列数据切片 - 修改为支持多步预测

        Args:
            data: [num_nodes, total_time, features]
            input_len: 输入序列长度（12）
            output_len: 输出序列长度（12）

        Returns:
            inputs: [num_samples, num_nodes, input_len, features]
            targets: [num_samples, num_nodes, output_len, features]
        """
        num_nodes, total_time, num_features = data.shape
        num_samples = total_time - input_len - output_len + 1

        inputs = []
        targets = []

        for i in range(num_samples):
            # 输入：第i到i+input_len步
            inputs.append(data[:, i:i+input_len, :])
            # 目标：第i+input_len到i+input_len+output_len步
            targets.append(data[:, i+input_len:i+input_len+output_len, :])

        # 堆叠成batch
        inputs = torch.stack(inputs, dim=0)   # [num_samples, num_nodes, input_len, features]
        targets = torch.stack(targets, dim=0) # [num_samples, num_nodes, output_len, features]

        return inputs, targets


# ----------------------
# 模型组件模块（添加数值稳定性）
# ----------------------
class ResidualConvBlock(nn.Module):
    """带短连接的残差卷积块"""
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
    """时空注意力机制"""
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.spatial_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.temporal_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mode: str = 'spatial') -> torch.Tensor:
        residual = x
        attn_fn = self.spatial_attn if mode == 'spatial' else self.temporal_attn
        x, _ = attn_fn(x, x, x)
        return self.norm(x + residual)

class MultiScaleTemporal(nn.Module):
    """多尺度时间特征提取（时间注意力池化版本）"""
    def __init__(self, hidden_dim: int, window_size: int = 12):
        super().__init__()
        self.window_size = window_size
        self.hidden_dim = hidden_dim

        # 多尺度卷积
        self.conv_blocks = nn.ModuleList([
            ResidualConvBlock(hidden_dim, hidden_dim, ks, ks//2)
            for ks in [3, 5, 11]
        ])

        # ✅ 时间注意力池化（替代flatten）
        self.temporal_attn = nn.Linear(hidden_dim, 1)

        # 融合层：3个尺度 × 64维 = 192维（而不是2304维）
        self.fusion = nn.Linear(3*hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*N, W, H]
        x = x.permute(0, 2, 1)  # [B*N, H, W]
        features = []

        for conv in self.conv_blocks:
            conv_out = conv(x)  # [B*N, H, W]

            # ✅ 时间注意力池化：学习每个时间步的重要性
            conv_out_t = conv_out.permute(0, 2, 1)  # [B*N, W, H]
            attn_scores = self.temporal_attn(conv_out_t)  # [B*N, W, 1]
            attn_weights = torch.softmax(attn_scores, dim=1)  # [B*N, W, 1]

            # 加权聚合所有时间步
            pooled = (conv_out_t * attn_weights).sum(dim=1)  # [B*N, H]
            features.append(pooled)

        # 融合3个尺度：[B*N, 3H]
        return self.fusion(torch.cat(features, dim=-1))

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
    """动态图结构生成器（增强数值稳定性）"""
    def __init__(self, num_nodes: int, feat_dim: int, hidden_dim: int):
        super().__init__()
        self.node_emb = nn.Parameter(torch.randn(num_nodes, hidden_dim) * 0.01)  # ✅ 降低初始化尺度
        self.fc = nn.Linear(feat_dim, hidden_dim)
        nn.init.xavier_uniform_(self.fc.weight, gain=0.5)  # ✅ 降低初始化增益

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, num_nodes, window, feat]
        x_agg = x.mean(dim=2)
        h = self.fc(x_agg) + self.node_emb.unsqueeze(0)

        # 数值稳定性保护
        h = torch.clamp(h, min=-10, max=10)  # 防止过大值
        similarity = torch.bmm(h, h.transpose(1, 2))
        similarity = torch.clamp(similarity, min=-20, max=20)  # 防止sigmoid饱和

        return torch.sigmoid(similarity)

class EnhancedDGSTAN(nn.Module):
    """增强型时空注意力网络（修复版）"""
    def __init__(self, num_nodes: int, feat_dim: int, adj_matrix: np.ndarray, config: ModelConfig):
        super().__init__()
        self.register_buffer('static_adj', torch.FloatTensor(adj_matrix))
        self.hidden_dim = config.hidden_dim

        # 图相关组件
        self.dynamic_graph = DynamicGraphGenerator(num_nodes, feat_dim, config.hidden_dim)
        self.graph_fusion = GraphFusionGate()

        # 时空特征提取
        self.spatial_proj = nn.Conv1d(feat_dim, config.hidden_dim, kernel_size=1)
        self.spatial_attn = SpatioTemporalAttention(config.hidden_dim, config.num_heads)
        self.temporal_net = MultiScaleTemporal(config.hidden_dim, config.seq_len)  # 传入window_size

        # 输出解码（修改为12步预测）
        self.output_layer = nn.Linear(config.hidden_dim, feat_dim * config.pred_len)  # 输出: hidden_dim -> 12
        self.pred_len = config.pred_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, window_size, _ = x.size()

        # 图结构融合
        dynamic_adj = self.dynamic_graph(x)
        adj = self.graph_fusion(self.static_adj, dynamic_adj)

        # 空间特征提取
        x = x.view(-1, window_size, x.size(3))
        x = self.spatial_proj(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = x.view(batch_size, num_nodes, window_size, self.hidden_dim)

        # 图卷积 + 残差
        x_permuted = x.permute(0, 2, 1, 3)
        residual = x_permuted.clone()

        # 修复：正确的图卷积einsum格式
        # adj: [B, N, N], x_permuted: [B, W, N, H]
        # 对每个时间步，用邻接矩阵聚合节点特征
        x_conv = torch.einsum('bnm,bwmh->bwnh', adj, x_permuted)
        x_conv = torch.clamp(x_conv, min=-10, max=10)  # ✅ 防止数值爆炸
        x_conv = x_conv + residual

        # 时空注意力
        x_attn = x_conv.reshape(batch_size * window_size, num_nodes, -1)
        x_attn = self.spatial_attn(x_attn, mode='spatial')
        x_attn = torch.clamp(x_attn, min=-10, max=10)  # ✅ 防止注意力输出过大
        x_attn = x_attn.view(batch_size, window_size, num_nodes, self.hidden_dim)

        # 时间特征提取
        x_time = x_attn.permute(0, 2, 1, 3)
        x_time = x_time.reshape(batch_size * num_nodes, window_size, -1)
        x_time = self.temporal_net(x_time)

        # 输出 - 修改为12步预测
        x_time = x_time.view(batch_size, num_nodes, self.hidden_dim)
        output = self.output_layer(x_time)  # [B, N, pred_len * feat_dim] = [B, N, 12*1]
        output = output.view(batch_size, num_nodes, self.pred_len, -1)  # [B, N, 12, 1]
        return output


# ----------------------
# 训练与评估模块（增强稳定性）
# ----------------------
class ModelTrainer:
    def __init__(self, model: nn.Module, config: ModelConfig, device: torch.device, train_loader: DataLoader):
        self.model = model.to(device)
        self.device = device
        self.config = config

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.base_lr,
            weight_decay=config.weight_decay
        )
        self.loss_fn = nn.MSELoss()
        self.scaler = GradScaler()

        self.steps_per_epoch = len(train_loader)
        self.total_steps = self.steps_per_epoch * config.epochs

        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.base_lr,
            total_steps=self.total_steps,
            pct_start=0.3,
            anneal_strategy='cos'
        )

    def train_epoch(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Tuple[float, Optional[float]]:
        """训练一个epoch（增强数值检查）"""
        self.model.train()
        total_loss = 0.0
        valid_batches = 0  # 记录有效batch数量

        for inputs, targets in train_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            with autocast():
                preds = self.model(inputs)
                loss = self.loss_fn(preds, targets)

            # 检查loss是否为NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️ 检测到NaN/Inf损失，跳过此batch")
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            # 增强梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            valid_batches += 1  # ✅ 计数有效batch

        avg_train_loss = total_loss / max(valid_batches, 1)  # ✅ 用实际batch数计算平均
        val_loss = None

        if val_loader:
            self.model.eval()
            total_val_loss = 0.0
            with torch.inference_mode():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    preds = self.model(inputs)
                    loss = self.loss_fn(preds, targets)

                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        total_val_loss += loss.item()

            val_loss = total_val_loss / len(val_loader)

        return avg_train_loss, val_loss

class ModelEvaluator:
    @staticmethod
    def evaluate(model: nn.Module, data_loader: DataLoader, mean: torch.Tensor,
                 std: torch.Tensor, device: torch.device) -> Tuple[float, float, float, float]:
        model.eval()
        preds, labels = [], []

        with torch.inference_mode():
            for inputs, targets in data_loader:
                inputs = inputs.to(device)
                preds.append(model(inputs).cpu())
                labels.append(targets.cpu())

        preds = torch.cat(preds) * std.cpu() + mean.cpu()
        labels = torch.cat(labels) * std.cpu() + mean.cpu()
        return ModelEvaluator.calculate_metrics(preds, labels)

    @staticmethod
    def calculate_metrics(preds: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float, float, float]:
        mae = F.l1_loss(preds, labels).item()
        rmse = torch.sqrt(F.mse_loss(preds, labels)).item()

        # MAPE with protection
        mask = labels > 10.0
        if mask.sum() > 0:
            mape = (torch.abs((labels[mask] - preds[mask]) / labels[mask]).mean() * 100).item()
        else:
            mape = 0.0

        ss_res = torch.sum((preds - labels)**2)
        ss_tot = torch.sum((labels - labels.mean())**2)
        r2 = (1 - ss_res / (ss_tot + 1e-6)).item()

        return mae, rmse, mape, r2

# ----------------------
# 主程序
# ----------------------
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = ModelConfig()
    print(f"训练设备：{device}")

    # 初始化数据
    adj_matrix, node_map = DataProcessor.build_adjacency_matrix("/data_ssd/other_models/数据集/PeMS04/PeMS04.csv")
    adj_norm = DataProcessor.normalize_adjacency(adj_matrix, config.sigma)
    pems_data, data_mean, data_std = DataProcessor.load_flow_with_train_stats(
        "/data_ssd/other_models/数据集/PeMS04/PeMS04.npz",
        config.train_ratio,
        device
    )

    # 准备数据集 - 修改为12步输入和12步输出
    input_seq, target_seq = DataProcessor.create_sequences(
        pems_data,
        config.seq_len,   # 输入长度：12步
        config.pred_len   # 输出长度：12步
    )
    # input_seq: [num_samples, num_nodes, 12, 1]
    # target_seq: [num_samples, num_nodes, 12, 1]
    input_seq = input_seq.cpu()
    target_seq = target_seq.cpu()

    total_samples = input_seq.size(0)
    train_size = int(total_samples * config.train_ratio)
    val_size = int(total_samples * config.val_ratio)

    train_dataset = TensorDataset(input_seq[:train_size], target_seq[:train_size])
    val_dataset = TensorDataset(input_seq[train_size:train_size+val_size],
                               target_seq[train_size:train_size+val_size])
    test_dataset = TensorDataset(input_seq[train_size+val_size:],
                                target_seq[train_size+val_size:])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    print(f"数据划分: 训练{len(train_dataset)} | 验证{len(val_dataset)} | 测试{len(test_dataset)}")

    # 初始化模型
    model = EnhancedDGSTAN(
        num_nodes=pems_data.shape[0],
        feat_dim=1,  # ✅ 只有flow
        adj_matrix=adj_norm,
        config=config
    )
    trainer = ModelTrainer(model, config, device, train_loader)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练循环
    print("\n" + "="*60)
    print("开始训练 DG-STAN...")
    print("="*60)

    best_val_loss = float('inf')
    for epoch in range(config.epochs):
        start_time = time.time()
        train_loss, val_loss = trainer.train_epoch(train_loader, val_loader)
        epoch_time = time.time() - start_time

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"dgstan_best.pth")

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{config.epochs}] {epoch_time:.1f}s | "
                  f"训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f}")

    # 加载最佳模型评估
    model.load_state_dict(torch.load("dgstan_best.pth"))

    print("\n" + "="*60)
    print("最终评估结果 (DG-STAN)")
    print("="*60)

    for name, loader in [("训练集", train_loader), ("验证集", val_loader), ("测试集", test_loader)]:
        mae, rmse, mape, r2 = ModelEvaluator.evaluate(model, loader, data_mean, data_std, device)
        print(f"{name:6s} | MAE: {mae:6.2f} | RMSE: {rmse:6.2f} | MAPE: {mape:5.2f}% | R²: {r2:.4f}")

