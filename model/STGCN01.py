# -*- coding: utf-8 -*-
"""
STGCN模型实现PEMS04交通流量预测
环境要求：Python 3.8+, PyTorch 1.10+, pandas, numpy
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ----------------------
# 步骤1：处理 pems.csv（调整邻接矩阵为0-1）
# ----------------------
def build_adjacency_matrix(csv_path):
    df = pd.read_csv(csv_path)
    all_nodes = sorted(set(df['from']).union(set(df['to'])))
    num_nodes = len(all_nodes)
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
    
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for _, row in df.iterrows():
        i = node_to_idx[row['from']]
        j = node_to_idx[row['to']]
        adj_matrix[i, j] = adj_matrix[j, i] = 1.0  # 使用0-1邻接矩阵
    
    print(f"总节点数: {num_nodes}")
    return adj_matrix, node_to_idx

adj_matrix, node_map = build_adjacency_matrix("E:\交通流预测研究\other_models\数据集\PeMS04\PeMS04.csv")

# ----------------------
# 步骤2：邻接矩阵归一化（对称归一化）
# ----------------------
def normalize_adjacency(adj_matrix):
    # 添加自环
    adj = adj_matrix + np.eye(adj_matrix.shape[0])
    # 计算度矩阵
    D = np.sum(adj, axis=1)
    D_sqrt_inv = np.diag(1.0 / np.sqrt(D))
    adj_normalized = D_sqrt_inv @ adj @ D_sqrt_inv
    return adj_normalized

adj_norm = normalize_adjacency(adj_matrix)

# ----------------------
# 步骤3：加载数据（保持原始处理）
# ----------------------
def load_pems_data(npz_path):
    data = np.load(npz_path)
    flow_data = data['data']  # [T, N, F]
    return torch.FloatTensor(flow_data).permute(1, 0, 2)  # [N, T, F]

pems_tensor = load_pems_data("PeMS04.npz")
print("原始数据形状: [节点数, 时间步长, 特征数]", pems_tensor.shape)

# ----------------------
# 步骤4：定义STGCN模型组件
# ----------------------
class TemporalConv(nn.Module):
    """时间卷积层"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size//2)
        )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: [batch, N, in_channels, T]
        x = x.permute(0, 2, 1, 3)  # [batch, in_channels, N, T]
        x = self.conv(x)
        x = x.permute(0, 2, 1, 3)  # [batch, N, out_channels, T]
        return self.relu(x)

class GraphConv(nn.Module):
    """图卷积层"""
    def __init__(self, in_channels, out_channels, adj_matrix):
        super().__init__()
        self.adj = torch.FloatTensor(adj_matrix)
        self.linear = nn.Linear(in_channels, out_channels)
        
    def forward(self, x):
        # x shape: [batch, N, in_channels, T]
        batch_size, num_nodes, in_channels, num_timesteps = x.shape
        
        # 空间特征聚合
        x = x.permute(0, 3, 1, 2)  # [batch, T, N, in_channels]
        x = torch.matmul(self.adj, x)  # [batch, T, N, in_channels]
        x = x.permute(0, 2, 3, 1)  # [batch, N, in_channels, T]
        
        # 线性变换
        x = self.linear(x.permute(0, 1, 3, 2))  # [batch, N, T, out_channels]
        return x.permute(0, 1, 3, 2)  # [batch, N, out_channels, T]

class STGCNBlock(nn.Module):
    """时空块（时间卷积 → 图卷积 → 时间卷积）"""
    def __init__(self, in_channels, out_channels, adj_matrix):
        super().__init__()
        self.temp_conv1 = TemporalConv(in_channels, out_channels)
        self.graph_conv = GraphConv(out_channels, out_channels, adj_matrix)
        self.temp_conv2 = TemporalConv(out_channels, out_channels)
        self.res_conv = nn.Conv2d(in_channels, out_channels, (1, 1)) if in_channels != out_channels else None
        
    def forward(self, x):
        residual = x
        x = self.temp_conv1(x)
        x = self.graph_conv(x)
        x = self.temp_conv2(x)
        if self.res_conv is not None:
            residual = self.res_conv(residual)
        x = x + residual  # 修复：改为非inplace加法
        return torch.relu(x)
class STGCN(nn.Module):
    """完整的STGCN模型"""
    def __init__(self, num_nodes, input_dim, adj_matrix, window_size, output_dim=3):  # 修改：默认输出维度设为3
        super().__init__()
        self.start_conv = TemporalConv(input_dim, 64)  # 初始通道变换
        
        # 两个时空块
        self.block1 = STGCNBlock(64, 64, adj_matrix)
        self.block2 = STGCNBlock(64, 64, adj_matrix)
        
        # 输出层
        self.fc = nn.Linear(64 * window_size, output_dim)
        
    def forward(self, x):
        # x shape: [batch, N, F, T]
        x = self.start_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        
        # 展平时间维度 [新增对维度顺序的注释]
        batch_size, num_nodes = x.shape[0], x.shape[1]
        x = x.permute(0, 1, 3, 2)  # [batch, N, T, C]
        x = x.reshape(batch_size, num_nodes, -1)
        return self.fc(x)

# ----------------------
# 步骤5：时间窗口处理（保持原始处理）
# ----------------------
def create_sequences(data, window_size=12):
    num_nodes, total_timesteps, num_features = data.shape
    
    sequences = []
    targets = []
    
    for i in range(total_timesteps - window_size):
        seq = data[:, i:i+window_size, :]  # [N, window, F]
        label = data[:, i+window_size, :]  # [N, F]
        sequences.append(seq)
        targets.append(label)
    
    return torch.stack(sequences, dim=1), torch.stack(targets, dim=1)

input_seq, target_seq = create_sequences(pems_tensor, window_size=12)
print("输入序列形状:", input_seq.shape, "目标形状:", target_seq.shape)

# ----------------------
# 步骤6：数据划分与加载（调整维度顺序）
# ----------------------
total_samples = input_seq.shape[1]
split_idx = int(total_samples * 0.8)

# 训练集
train_input = input_seq[:, :split_idx, :, :]
train_target = target_seq[:, :split_idx, :]

# 测试集
test_input = input_seq[:, split_idx:, :, :]
test_target = target_seq[:, split_idx:, :]

# 调整维度顺序为 [samples, N, F, window]
train_dataset = TensorDataset(
    train_input.permute(1, 0, 3, 2),  # [samples, N, F, window]
    train_target.permute(1, 0, 2)
)
test_dataset = TensorDataset(
    test_input.permute(1, 0, 3, 2),
    test_target.permute(1, 0, 2)
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# ----------------------
# 步骤7：初始化模型
# ----------------------
model = STGCN(
    num_nodes=pems_tensor.shape[0],
    input_dim=pems_tensor.shape[-1],
    adj_matrix=adj_norm,
    window_size=12,
    output_dim=pems_tensor.shape[-1]
)
print(model)

# ----------------------
# 训练循环（保持原始逻辑）
# ----------------------
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(3):
    model.train()
    total_loss = 0
    for batch_inputs, batch_targets in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = loss_fn(outputs, batch_targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "stgcn_model.pth")

# ----------------------
# 评估指标计算（添加R²）
# ----------------------
def calculate_metrics(preds, labels):
    epsilon = 1e-6
    mae = torch.abs(preds - labels).mean().item()
    rmse = torch.sqrt(torch.mean((preds - labels)**2)).item()
    mape = torch.mean(torch.abs((labels - preds) / (labels + epsilon)) * 100).item()
    
    # R²计算
    ss_res = torch.sum((labels - preds) ** 2)
    ss_tot = torch.sum((labels - torch.mean(labels)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + epsilon)).item()
    
    return mae, rmse, mape, r2

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        all_preds.append(outputs)
        all_labels.append(targets)
        
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

mae, rmse, mape, r2 = calculate_metrics(all_preds, all_labels)

print("\n评估结果：")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"R² Score: {r2:.4f}")  # 新增输出行