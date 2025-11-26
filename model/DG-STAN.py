# -*- coding: utf-8 -*-
"""
DG-STAN 14.6版（新增测试集模块）
主要改进：
1. 增加测试集划分及评估
2. 优化数据划分策略
3. 训练/验证/测试结果独立报告
"""
from typing import Tuple, Dict, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
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
        self.pred_len = 1         # 预测长度
        self.hidden_dim = 64      # 隐藏层维度
        self.num_heads = 4        # 注意力头数
        self.batch_size = 32      # 批量大小
        self.base_lr = 1e-3       # 基准学习率
        self.weight_decay = 1e-4  # 权重衰减
        self.epochs = 10          # 训练轮数
        self.sigma = 100.0        # 邻接矩阵高斯核标准差
        self.train_ratio = 0.7    # 新增：训练集比例
        self.val_ratio = 0.1      # 新增：验证集比例

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
        
        # 向量化构造邻接矩阵
        from_idx = df['from'].map(node_map).values.astype(int)
        to_idx = df['to'].map(node_map).values.astype(int)
        cost = df['cost'].values.astype(np.float32)
        
        adj_matrix = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
        adj_matrix[from_idx, to_idx] = cost
        adj_matrix[to_idx, from_idx] = cost  # 对称填充
        
        print(f"构建邻接矩阵完成，节点数：{len(nodes)}，示例距离(73->5)：{adj_matrix[node_map[73], node_map[5]]:.1f}")
        return adj_matrix, node_map

    @staticmethod
    def normalize_adjacency(adj_matrix: np.ndarray, sigma: float) -> np.ndarray:
        """邻接矩阵归一化处理"""
        adj_normalized = np.exp(-(adj_matrix ** 2) / (2 * sigma ** 2))
        np.fill_diagonal(adj_normalized, 1)
        return adj_normalized

    @staticmethod
    def load_pems_data(npz_path: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """交通流量数据加载与标准化"""
        data = np.load(npz_path)['data']  # [T, N, F]
        
        # 数据规范化
        mean = data.mean(axis=(0, 1), keepdims=True)
        std = data.std(axis=(0, 1), keepdims=True) + 1e-5
        normalized_data = (data - mean) / std
        
        return (
            torch.FloatTensor(normalized_data).permute(1, 0, 2).to(device),
            torch.FloatTensor(mean.squeeze()),
            torch.FloatTensor(std.squeeze())
        )

    @staticmethod
    def create_sequences(data: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """时间序列数据切片"""
        _, total_time, _ = data.shape
        indices = torch.arange(window_size)[None, :] + torch.arange(total_time - window_size)[:, None]
        return data[:, indices], data[:, window_size:]


# ----------------------
# 模型组件模块
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
    """多尺度时间特征提取"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.conv_blocks = nn.ModuleList([
            ResidualConvBlock(hidden_dim, hidden_dim, ks, ks//2)
            for ks in [3, 5, 11]
        ])
        self.fusion = nn.Linear(3*hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入维度：[batch*num_nodes, seq_len, features]
        x = x.permute(0, 2, 1)  # [B*N, features, seq_len]
        features = [conv(x)[:, :, -1] for conv in self.conv_blocks]  # 提取末端特征
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
    """动态图结构生成器"""
    def __init__(self, num_nodes: int, feat_dim: int, hidden_dim: int):
        super().__init__()
        self.node_emb = nn.Parameter(torch.randn(num_nodes, hidden_dim))
        self.fc = nn.Linear(feat_dim, hidden_dim)
        nn.init.normal_(self.node_emb, std=0.01)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, num_nodes, window, feat]
        x_agg = x.mean(dim=2)  # 时间维度聚合
        h = self.fc(x_agg) + self.node_emb.unsqueeze(0)
        return torch.sigmoid(torch.bmm(h, h.transpose(1, 2)))

class EnhancedDGSTAN(nn.Module):
    """增强型时空注意力网络"""
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
        self.temporal_net = MultiScaleTemporal(config.hidden_dim)
        
        # 输出解码
        self.output_layer = nn.Linear(config.hidden_dim, feat_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入维度：[batch_size, num_nodes, window, feat]
        batch_size, num_nodes, window_size, _ = x.size()
        
        # 图结构融合
        dynamic_adj = self.dynamic_graph(x)  # [B, N, N]
        adj = self.graph_fusion(self.static_adj, dynamic_adj)  # [B, N, N]
        
        # 空间特征提取
        x = x.view(-1, window_size, x.size(3))  # [B*N, window, feat]
        x = self.spatial_proj(x.permute(0, 2, 1))  # [B*N, hidden_dim, window]
        x = x.permute(0, 2, 1)  # [B*N, window, hidden_dim]
        x = x.view(batch_size, num_nodes, window_size, self.hidden_dim)  # [B, N, window, H]
        
        # 调整维度顺序为 [B, window, N, H] 用于图卷积
        x_permuted = x.permute(0, 2, 1, 3)  # [B, W, N, H]
        residual = x_permuted.clone()  # 残差保留原特征
        
        # 动态图卷积 (使用正确的einsum维度)
        x_conv = torch.einsum('bij,bwnh->bwih', adj, x_permuted)
        
        # 残差连接
        x_conv = x_conv + residual  # 形状保持 [B, W, N, H]
        
        # 时空注意力处理
        x_attn = x_conv.reshape(batch_size * window_size, num_nodes, -1)
        x_attn = self.spatial_attn(x_attn, mode='spatial')  # [B*W, N, H]
        x_attn = x_attn.view(batch_size, window_size, num_nodes, self.hidden_dim)
        
        # 时间特征提取
        x_time = x_attn.permute(0, 2, 1, 3)  # [B, N, W, H]
        x_time = x_time.reshape(batch_size * num_nodes, window_size, -1)
        x_time = self.temporal_net(x_time)  # [B*N, H]
        
        # 输出层
        output = x_time.view(batch_size, num_nodes, self.hidden_dim)
        return self.output_layer(output)


# ----------------------
# 训练与评估模块
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
        """新增验证集评估功能"""
        self.model.train()
        total_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            with autocast(device_type='cuda', dtype=torch.float32):
                preds = self.model(inputs)
                loss = self.loss_fn(preds, targets)
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        val_loss = None
        
        if val_loader:
            self.model.eval()
            total_val_loss = 0.0
            with torch.inference_mode():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    preds = self.model(inputs)
                    total_val_loss += self.loss_fn(preds, targets).item()
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
                
        preds = torch.cat(preds) * std + mean
        labels = torch.cat(labels) * std + mean
        return ModelEvaluator.calculate_metrics(preds, labels)

    @staticmethod
    def calculate_metrics(preds: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float, float, float]:
        epsilon = 1e-6
        mae = F.l1_loss(preds, labels).item()
        rmse = torch.sqrt(F.mse_loss(preds, labels)).item()
        mape = (torch.abs((labels - preds) / (labels + epsilon)).mean() * 100).item()
        ss_res = torch.sum((preds - labels)**2)
        ss_tot = torch.sum((labels - labels.mean())**2)
        r2 = 1 - ss_res / (ss_tot + epsilon)
        return mae, rmse, mape, r2.item()

# ----------------------
# 主程序（包含测试集评估）
# ----------------------
if __name__ == "__main__":
    # 环境配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = ModelConfig()
    print(f"当前训练设备：{device}")
    
    # 初始化数据
    adj_matrix, node_map = DataProcessor.build_adjacency_matrix("PeMS04.csv")
    adj_norm = DataProcessor.normalize_adjacency(adj_matrix, config.sigma)
    pems_data, data_mean, data_std = DataProcessor.load_pems_data("PeMS04.npz", device)
    
    # 准备数据集（新增三划分逻辑）
    input_seq, target_seq = DataProcessor.create_sequences(pems_data, config.seq_len)
    
    # 维度调整 [样本数, 节点数, 序列长度, 特征数]
    input_seq = input_seq.permute(1, 0, 2, 3).cpu()
    target_seq = target_seq.permute(1, 0, 2).cpu()
    
    # 计算各数据集大小
    total_samples = input_seq.size(0)
    train_size = int(total_samples * config.train_ratio)
    val_size = int(total_samples * config.val_ratio)
    test_size = total_samples - train_size - val_size
    
    # 划分数据集
    train_dataset = TensorDataset(input_seq[:train_size], target_seq[:train_size])
    val_dataset = TensorDataset(input_seq[train_size:train_size+val_size], 
                               target_seq[train_size:train_size+val_size])
    test_dataset = TensorDataset(input_seq[train_size+val_size:], 
                                target_seq[train_size+val_size:])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    
    # 初始化模型与训练器
    model = EnhancedDGSTAN(
        num_nodes=pems_data.shape[0],
        feat_dim=pems_data.shape[-1],
        adj_matrix=adj_norm,
        config=config
    )
    trainer = ModelTrainer(model, config, device, train_loader)
    
    # 训练循环（新增验证功能）
    print("开始训练...")
    best_val_loss = float('inf')
    for epoch in range(config.epochs):
        start_time = time.time()
        train_loss, val_loss = trainer.train_epoch(train_loader, val_loader)
        epoch_time = time.time() - start_time
        
        # 模型保存逻辑（可选）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"best_model_epoch{epoch+1}.pth")
        
        print(f"Epoch [{epoch+1}/{config.epochs}] 耗时 {epoch_time:.1f}s")
        print(f"├─ 训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f}")

    # 完整评估（新增测试集评估）
    print("\n模型综合评估:")
    for name, loader in [("训练集", train_loader), ("验证集", val_loader), ("测试集", test_loader)]:
        print(f"正在评估 {name}...")
        mae, rmse, mape, r2 = ModelEvaluator.evaluate(model, loader, data_mean, data_std, device)
        print(f"│ {name}结果：MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape:.2f}% | R²: {r2:.4f}")

    # 模型保存
    torch.save({
        'model_state': model.state_dict(),
        'adj_matrix': adj_matrix,
        'node_mapping': node_map,
        'config': config.__dict__
    }, "dgstan_v14_with_test.pth")
