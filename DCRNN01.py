# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
# ----------------------
# 步骤1：处理 pems.csv（有向图修正）
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
        adj_matrix[i, j] = row['cost']
    
    print(f"总节点数: {num_nodes}, 示例距离(73->5): {adj_matrix[node_to_idx[73], node_to_idx[5]]:.1f}")
    return adj_matrix, node_to_idx
adj_matrix, node_map = build_adjacency_matrix("E:\交通流预测研究\other_models\数据集\PeMS04\PeMS04.csv")
# ----------------------
# 步骤2：生成两种转移矩阵（正向和反向）并转换为张量
# ----------------------
def normalize_adjacency(adj_matrix, sigma=100.0):
    def gaussian_kernel(matrix):
        return np.exp(-(matrix ** 2) / (2 * sigma ** 2))
    
    forward_adj = gaussian_kernel(adj_matrix)
    backward_adj = gaussian_kernel(adj_matrix.T)
    
    np.fill_diagonal(forward_adj, 1)
    np.fill_diagonal(backward_adj, 1)
    
    forward_adj = forward_adj / forward_adj.sum(axis=1, keepdims=True)
    backward_adj = backward_adj / backward_adj.sum(axis=1, keepdims=True)
    
    return [forward_adj, backward_adj]
supports = normalize_adjacency(adj_matrix)
# 将numpy数组转换为PyTorch张量
support_tensors = [torch.from_numpy(s).float() for s in supports]
# ----------------------
# 步骤3：加载数据（维度修复）
# ----------------------
def load_pems_data(npz_path):
    data = np.load(npz_path)
    flow_data = data['data']
    return torch.FloatTensor(flow_data).permute(1, 0, 2)  # [N, T, F]
pems_tensor = load_pems_data("E:\交通流预测研究\other_models\数据集\PeMS04\PeMS04.npz")
print("原始数据形状: [节点数, 时间步长, 特征数]", pems_tensor.shape)
# ----------------------
# DCRNN组件定义
# ----------------------
class DiffusionGraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, supports):
        super().__init__()
        self.supports = supports
        self.num_supports = len(supports)
        self.weights = nn.Parameter(torch.FloatTensor(input_dim * self.num_supports, output_dim))
        self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        nn.init.xavier_normal_(self.weights)
        nn.init.zeros_(self.bias)
        
    def forward(self, x):
        batch_size, num_nodes, input_dim = x.shape
        support_outputs = []
        for support in self.supports:
            support = support.to(x.device)
            expanded_support = support.unsqueeze(0).expand(batch_size, -1, -1)
            output = torch.bmm(expanded_support, x)
            support_outputs.append(output)
        
        support_cat = torch.cat(support_outputs, dim=-1)
        flattened = support_cat.reshape(batch_size * num_nodes, -1)
        output = torch.matmul(flattened, self.weights) + self.bias
        output = output.reshape(batch_size, num_nodes, -1)
        return output

class DCGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, supports):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gates_conv = DiffusionGraphConv(input_dim + hidden_dim, 2 * hidden_dim, supports)
        self.candidate_conv = DiffusionGraphConv(input_dim + hidden_dim, hidden_dim, supports)
        
    def forward(self, x, h):
        combined = torch.cat([x, h], dim=-1)
        gates = torch.sigmoid(self.gates_conv(combined))
        reset_gate, update_gate = torch.split(gates, self.hidden_dim, dim=-1)
        
        combined_reset = torch.cat([x, reset_gate * h], dim=-1)
        candidate = torch.tanh(self.candidate_conv(combined_reset))
        
        new_h = update_gate * h + (1 - update_gate) * candidate
        return new_h

class DCRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, supports):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dcgru_cells = nn.ModuleList(
            [DCGRUCell(input_dim if i==0 else hidden_dim, hidden_dim, supports)
             for i in range(num_layers)]
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        batch_size, seq_len, num_nodes, input_dim = x.shape
        h = torch.zeros(batch_size, num_nodes, self.hidden_dim).to(x.device)
        
        for layer in self.dcgru_cells:
            layer_h = []
            for t in range(seq_len):
                h = layer(x[:, t, :, :], h)
                layer_h.append(h)
            x = torch.stack(layer_h, dim=1)
        
        return self.output_layer(x[:, -1, :, :])
# ----------------------
# 步骤4：创建DCRNN模型（使用修正后的support_tensors）
# ----------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DCRNN(
    input_dim=pems_tensor.shape[-1],
    hidden_dim=64,
    num_layers=2,
    supports=support_tensors  # 使用已转换为张量的支持矩阵
).to(device)
print(model)

# ----------------------
# 步骤5：时间窗口处理（保持DATASET维度一致）
# ----------------------
def create_sequences(data, window_size=12):
    num_nodes, total_timesteps, num_features = data.shape
    sequences, labels = [], []
    
    for i in range(total_timesteps - window_size):
        seq = data[:, i:i+window_size, :]  # [N, window, F]
        label = data[:, i+window_size, :] 
        sequences.append(seq)
        labels.append(label)
    
    input_seq = torch.stack(sequences, dim=1)  # [N, samples, window, F]
    target_seq = torch.stack(labels, dim=1)      # [N, samples, F]
    return input_seq, target_seq

input_seq, target_seq = create_sequences(pems_tensor)
print("输入序列形状:", input_seq.shape, "目标形状:", target_seq.shape)

# ----------------------
# 步骤6：数据加载器调整（适配DCRNN输入）
# ----------------------
total_samples = input_seq.shape[1]
split_idx = int(total_samples * 0.8)

train_input = input_seq[:, :split_idx].permute(1, 2, 0, 3)  # [samples, window, N, F]
train_target = target_seq[:, :split_idx].permute(1, 0, 2)   # [samples, N, F]

test_input = input_seq[:, split_idx:].permute(1, 2, 0, 3)
test_target = target_seq[:, split_idx:].permute(1, 0, 2)

train_dataset = TensorDataset(train_input, train_target)
test_dataset = TensorDataset(test_input, test_target)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# ----------------------
# 训练配置
# ----------------------
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# ----------------------
# 训练循环
# ----------------------
for epoch in range(10):
    model.train()
    total_loss = 0
    for batch_inputs, batch_targets in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_inputs.to(device))
        loss = loss_fn(outputs, batch_targets.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# ----------------------
# 序列predictron及评估
# ----------------------
# ----------------------
# 序列predictron及评估
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
all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, targets in test_loader:
        preds = model(inputs.to(device))
        all_preds.append(preds.cpu())
        all_labels.append(targets)

all_preds = torch.cat(all_preds, dim=0)
all_labels = torch.cat(all_labels, dim=0)

mae, rmse, mape, r2 = calculate_metrics(all_preds, all_labels)
print(f"\n评估结果:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"R² Score: {r2:.4f}")