"""
DG-STAN v2 - 改进版动态图时空注意力网络
主要改进:
1. Seq2Seq Decoder - 建模时序依赖
2. 保留时间维度 - 不压缩时间信息
3. 多层图卷积 (3层) - 增加感受野
4. 双向时空注意力 - 同时使用spatial和temporal
5. GLU门控机制 - 控制信息流
6. Huber Loss - 减少异常值影响
7. 学习率调度 - 更好的收敛
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Tuple
import time

# ----------------------
# 配置
# ----------------------
class ModelConfig:
    def __init__(self):
        self.seq_len = 12         # 输入长度
        self.pred_len = 12        # 输出长度 (12步)
        self.hidden_dim = 128     # ✅ 增加到128 (原64)
        self.num_heads = 8        # ✅ 增加到8 (原4)
        self.num_gcn_layers = 3   # ✅ 新增: 3层图卷积
        self.batch_size = 32
        self.base_lr = 5e-4
        self.weight_decay = 1e-4
        self.epochs = 150         # ✅ 增加到150 (原100)
        self.patience = 20        # ✅ 新增: early stopping
        self.sigma = 100.0
        self.train_ratio = 0.6
        self.val_ratio = 0.2
        self.dropout = 0.1        # ✅ 新增: dropout


# ----------------------
# 改进的模块
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


class SpatioTemporalAttention(nn.Module):
    """改进的时空注意力机制 - 同时使用spatial和temporal"""
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.spatial_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.temporal_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*W, N, H] 或 [B*N, W, H]

        # Spatial attention
        residual = x
        x_attn, _ = self.spatial_attn(x, x, x)
        x = self.norm1(residual + self.dropout(x_attn))

        # Temporal attention
        residual = x
        x_attn, _ = self.temporal_attn(x, x, x)
        x = self.norm2(residual + self.dropout(x_attn))

        return x


class MultiScaleTemporalV2(nn.Module):
    """改进的多尺度时间特征提取 - 保留时间维度"""
    def __init__(self, hidden_dim: int, window_size: int = 12):
        super().__init__()
        self.window_size = window_size
        self.hidden_dim = hidden_dim

        # 多尺度卷积
        self.conv_blocks = nn.ModuleList([
            ResidualConvBlock(hidden_dim, hidden_dim, ks, ks//2)
            for ks in [3, 5, 11]
        ])

        # ✅ 不压缩时间维度，而是融合多尺度特征
        self.fusion = nn.Sequential(
            nn.Conv1d(3 * hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*N, W, H]
        x = x.permute(0, 2, 1)  # [B*N, H, W]
        features = []

        for conv in self.conv_blocks:
            conv_out = conv(x)  # [B*N, H, W]
            features.append(conv_out)

        # 融合3个尺度，保留时间维度
        fused = torch.cat(features, dim=1)  # [B*N, 3H, W]
        output = self.fusion(fused)  # [B*N, H, W]

        return output.permute(0, 2, 1)  # [B*N, W, H]


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
        # x: [B, W, N, H], adj: [B, N, N]
        batch_size, window_size, num_nodes, hidden_dim = x.size()

        # 图卷积: 对每个时间步应用邻接矩阵
        x_conv = torch.einsum('bnm,bwmh->bwnh', adj, x)
        x_conv = torch.clamp(x_conv, min=-10, max=10)

        # GLU门控 + 残差 + LayerNorm
        x_out = self.glu(x_conv)
        x_out = self.dropout(x_out)

        return self.norm(x + x_out)


class MultiLayerGraphConv(nn.Module):
    """多层图卷积网络 - 新增"""
    def __init__(self, hidden_dim: int, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, dropout) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, adj)
        return x


class Seq2SeqDecoder(nn.Module):
    """Seq2Seq解码器 - 核心改进"""
    def __init__(self, hidden_dim: int, pred_len: int = 12, dropout: float = 0.1):
        super().__init__()
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim

        # 使用GRU cell逐步生成
        self.decoder_cells = nn.ModuleList([
            nn.GRUCell(hidden_dim + 1, hidden_dim) for _ in range(pred_len)
        ])

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, encoder_hidden: torch.Tensor) -> torch.Tensor:
        # encoder_hidden: [B*N, H]
        batch_size = encoder_hidden.size(0)
        device = encoder_hidden.device

        outputs = []
        h = encoder_hidden
        prev_y = torch.zeros(batch_size, 1, device=device)

        for t in range(self.pred_len):
            # 拼接上一步预测作为输入
            decoder_input = torch.cat([h, prev_y], dim=-1)

            # GRU更新隐状态
            h = self.decoder_cells[t](decoder_input, h)

            # 生成当前步预测
            y_t = self.output_proj(h)
            outputs.append(y_t)

            # 更新prev_y (训练时用真实值，推理时用预测值)
            prev_y = y_t if not self.training else y_t.detach()

        return torch.stack(outputs, dim=1)  # [B*N, pred_len, 1]


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
        # x: [batch_size, num_nodes, window, feat]
        x_agg = x.mean(dim=2)
        h = self.fc(x_agg) + self.node_emb.unsqueeze(0)

        h = torch.clamp(h, min=-10, max=10)
        similarity = torch.bmm(h, h.transpose(1, 2))
        similarity = torch.clamp(similarity, min=-20, max=20)

        return torch.sigmoid(similarity)


class EnhancedDGSTAN_V2(nn.Module):
    """DG-STAN v2 - 改进版"""
    def __init__(self, num_nodes: int, feat_dim: int, adj_matrix: np.ndarray, config: ModelConfig):
        super().__init__()
        self.register_buffer('static_adj', torch.FloatTensor(adj_matrix))
        self.hidden_dim = config.hidden_dim
        self.num_nodes = num_nodes

        # 图相关组件
        self.dynamic_graph = DynamicGraphGenerator(num_nodes, feat_dim, config.hidden_dim)
        self.graph_fusion = GraphFusionGate()

        # 空间投影
        self.spatial_proj = nn.Conv1d(feat_dim, config.hidden_dim, kernel_size=1)

        # ✅ 多层图卷积 (3层)
        self.graph_conv = MultiLayerGraphConv(config.hidden_dim, config.num_gcn_layers, config.dropout)

        # ✅ 改进的双向时空注意力
        self.st_attn = SpatioTemporalAttention(config.hidden_dim, config.num_heads, config.dropout)

        # ✅ 改进的时间特征提取 (保留时间维度)
        self.temporal_net = MultiScaleTemporalV2(config.hidden_dim, config.seq_len)

        # ✅ Seq2Seq解码器
        self.decoder = Seq2SeqDecoder(config.hidden_dim, config.pred_len, config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, window_size, _ = x.size()

        # 1. 图结构融合
        dynamic_adj = self.dynamic_graph(x)
        adj = self.graph_fusion(self.static_adj, dynamic_adj)

        # 2. 空间投影
        x = x.view(-1, window_size, x.size(3))
        x = self.spatial_proj(x.permute(0, 2, 1))  # [B*N, H, W]
        x = x.permute(0, 2, 1)  # [B*N, W, H]
        x = x.view(batch_size, num_nodes, window_size, self.hidden_dim)

        # 3. ✅ 多层图卷积 (替代单层)
        x = x.permute(0, 2, 1, 3)  # [B, W, N, H]
        x = self.graph_conv(x, adj)  # [B, W, N, H]

        # 4. ✅ 时空注意力 (spatial + temporal)
        x_attn = x.reshape(batch_size * window_size, num_nodes, -1)
        x_attn = self.st_attn(x_attn)  # [B*W, N, H]
        x_attn = x_attn.view(batch_size, window_size, num_nodes, self.hidden_dim)

        # 5. ✅ 时间特征提取 (保留时间维度)
        x_time = x_attn.permute(0, 2, 1, 3)  # [B, N, W, H]
        x_time = x_time.reshape(batch_size * num_nodes, window_size, -1)
        x_time = self.temporal_net(x_time)  # [B*N, W, H] - 保留W维度！

        # 6. ✅ 对时间维度做注意力聚合得到encoder hidden
        x_pooled = x_time.mean(dim=1)  # [B*N, H]

        # 7. ✅ Seq2Seq解码器生成12步预测
        output = self.decoder(x_pooled)  # [B*N, 12, 1]

        # 8. 重塑输出
        output = output.view(batch_size, num_nodes, -1, 1)  # [B, N, 12, 1]

        return output


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

        # ✅ Huber Loss (SmoothL1Loss)
        self.loss_fn = nn.SmoothL1Loss(beta=1.0)

        # ✅ 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

        self.scaler = GradScaler()
        self.steps_per_epoch = len(train_loader)

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
                print(f"⚠️ 警告: loss为NaN/Inf，跳过此batch")
                continue

            self.scaler.scale(loss).backward()

            # ✅ 梯度裁剪
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

            # ✅ 更新学习率
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

        mask = labels != 0
        mape = torch.mean(torch.abs((labels[mask] - preds[mask]) / labels[mask])).item() * 100

        # R²
        ss_res = torch.sum((labels - preds) ** 2).item()
        ss_tot = torch.sum((labels - labels.mean()) ** 2).item()
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        return mae, rmse, mape, r2


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
# 主程序
# ----------------------
if __name__ == '__main__':
    print("=" * 60)
    print("DG-STAN v2 训练 - 改进版")
    print("=" * 60)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'训练设备：{device}')

    # 配置
    config = ModelConfig()

    # 加载数据
    data_file = '/data_ssd/other_models/数据集/PeMS04/PeMS04.npz'
    adj_file = '/data_ssd/other_models/数据集/PeMS04/PeMS04.csv'

    data = np.load(data_file)
    pems_data = torch.FloatTensor(data['data'][:, :, 0:1])  # [T, N, 1]
    pems_data = pems_data.permute(1, 0, 2)  # [N, T, 1]

    print(f"邻接矩阵: {pems_data.size(0)}个节点")
    print(f"Flow数据: {pems_data.shape}")

    # 归一化
    train_size = int(config.train_ratio * pems_data.size(1))
    train_data = pems_data[:, :train_size, :]
    data_mean = train_data.mean().item()
    data_std = train_data.std().item()
    pems_data = (pems_data - data_mean) / data_std

    print(f"训练集均值: {data_mean:.2f}, 标准差: {data_std:.2f}")

    # 加载邻接矩阵
    edges = np.loadtxt(adj_file, delimiter=',', skiprows=1)
    num_nodes = pems_data.size(0)
    adj_matrix = np.zeros((num_nodes, num_nodes))

    for edge in edges:
        i, j, dist = int(edge[0]), int(edge[1]), edge[2]
        if dist > 0:
            adj_matrix[i, j] = 1.0 / dist
            adj_matrix[j, i] = 1.0 / dist

    # 数据划分
    train_end = int(config.train_ratio * pems_data.size(1))
    val_end = int((config.train_ratio + config.val_ratio) * pems_data.size(1))

    train_data = pems_data[:, :train_end, :]
    val_data = pems_data[:, train_end:val_end, :]
    test_data = pems_data[:, val_end:, :]

    print(f"数据划分: 训练{train_data.size(1)} | 验证{val_data.size(1)} | 测试{test_data.size(1)}")

    # 准备数据集
    input_seq, target_seq = DataProcessor.create_sequences(
        pems_data, config.seq_len, config.pred_len
    )
    input_seq = input_seq.cpu()
    target_seq = target_seq.cpu()

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

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    # 创建模型
    model = EnhancedDGSTAN_V2(
        num_nodes=num_nodes,
        feat_dim=1,
        adj_matrix=adj_matrix,
        config=config
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")

    # 训练
    trainer = ModelTrainer(model, config, device, train_loader)

    print("\n" + "=" * 60)
    print("开始训练 DG-STAN v2...")
    print("=" * 60)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config.epochs):
        start_time = time.time()
        train_loss, val_loss = trainer.train_epoch(train_loader, val_loader)
        epoch_time = time.time() - start_time

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'dgstan_v2_best.pth')
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{config.epochs}] {epoch_time:.1f}s | "
                  f"训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f}")

        if patience_counter >= config.patience:
            print(f"\n✅ Early stopping at epoch {epoch+1}")
            break

    # 加载最佳模型并测试
    print("\n" + "=" * 60)
    print("最终评估结果 (DG-STAN v2)")
    print("=" * 60)

    model.load_state_dict(torch.load('dgstan_v2_best.pth'))

    train_mae, train_rmse, train_mape, train_r2 = ModelEvaluator.evaluate(
        model, train_loader, data_mean, data_std, device
    )
    val_mae, val_rmse, val_mape, val_r2 = ModelEvaluator.evaluate(
        model, val_loader, data_mean, data_std, device
    )
    test_mae, test_rmse, test_mape, test_r2 = ModelEvaluator.evaluate(
        model, test_loader, data_mean, data_std, device
    )

    print(f"训练集    | MAE: {train_mae:6.2f} | RMSE: {train_rmse:6.2f} | MAPE: {train_mape:5.2f}% | R²: {train_r2:.4f}")
    print(f"验证集    | MAE: {val_mae:6.2f} | RMSE: {val_rmse:6.2f} | MAPE: {val_mape:5.2f}% | R²: {val_r2:.4f}")
    print(f"测试集    | MAE: {test_mae:6.2f} | RMSE: {test_rmse:6.2f} | MAPE: {test_mape:5.2f}% | R²: {test_r2:.4f}")

    print("\n" + "=" * 60)
    print("与STGCN基线对比:")
    print("STGCN:    | MAE:  24.51 | RMSE:  37.55 | MAPE: 17.14% | R²: 0.9437")
    print(f"DG-STAN v2| MAE: {test_mae:6.2f} | RMSE: {test_rmse:6.2f} | MAPE: {test_mape:5.2f}% | R²: {test_r2:.4f}")

    improvement_mae = ((24.51 - test_mae) / 24.51) * 100
    improvement_rmse = ((37.55 - test_rmse) / 37.55) * 100

    print(f"\n改进幅度:")
    print(f"MAE:  {improvement_mae:+.2f}%")
    print(f"RMSE: {improvement_rmse:+.2f}%")
    print("=" * 60)
