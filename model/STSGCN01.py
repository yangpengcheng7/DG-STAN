# # -*- coding: utf-8 -*-
# """
# 替换DGSTAN为STSGCN后的完整代码
# 环境要求：Python 3.8+, PyTorch 1.10+, pandas, numpy
# """
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset
#
# # ----------------------
# # 步骤1：处理 distance.csv
# # ----------------------
# def build_adjacency_matrix(csv_path):
#     df = pd.read_csv(csv_path)
#     all_nodes = sorted(set(df['from']).union(set(df['to'])))
#     num_nodes = len(all_nodes)
#     node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
#
#     adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
#     for _, row in df.iterrows():
#         i = node_to_idx[row['from']]
#         j = node_to_idx[row['to']]
#         adj_matrix[i, j] = adj_matrix[j, i] = row['cost']
#
#     print(f"总节点数: {num_nodes}, 示例距离(73->5): {adj_matrix[node_to_idx[73], node_to_idx[5]]:.1f}")
#     return adj_matrix, node_to_idx
#
# adj_matrix, node_map = build_adjacency_matrix("E:\交通流预测研究\other_models\数据集\PeMS04\PeMS04.csv")
#
# # ----------------------
# # 步骤2：邻接矩阵归一化（空间部分）
# # ----------------------
# def normalize_adjacency(adj):
#     sigma = 100.0
#     adj_normalized = np.exp(-(adj ** 2) / (2 * sigma ** 2))
#     np.fill_diagonal(adj_normalized, 1)
#     return adj_normalized
#
# adj_norm = normalize_adjacency(adj_matrix)
#
# # ----------------------
# # 新增步骤：构造时空联合邻接矩阵
# # ----------------------
# def build_spatiotemporal_adj(space_adj, window_size):
#     num_nodes = space_adj.shape[0]
#     total = num_nodes * window_size
#     adj_ext = np.zeros((total, total), dtype=np.float32)
#
#     # 填充每个时间步的空间邻接
#     for t in range(window_size):
#         start = t * num_nodes
#         end = start + num_nodes
#         adj_ext[start:end, start:end] = space_adj
#
#     # 填充时间邻接（相邻时间步同一节点）
#     for t in range(window_size - 1):
#         for i in range(num_nodes):
#             idx1 = t * num_nodes + i
#             idx2 = (t + 1) * num_nodes + i
#             adj_ext[idx1, idx2] = adj_ext[idx2, idx1] = 1.0
#
#     return adj_ext
#
# window_size = 12  # 与后续create_sequences中参数一致
# adj_ext = build_spatiotemporal_adj(adj_norm, window_size=window_size)
# adj_ext = torch.FloatTensor(adj_ext)
#
# # ----------------------
# # 步骤3：加载数据
# # ----------------------
# def load_pems_data(npz_path):
#     data = np.load(npz_path)
#     flow_data = data['data']  # [T, N, F]
#     return torch.FloatTensor(flow_data).permute(1, 0, 2)  # [N, T, F]
#
# pems_tensor = load_pems_data("E:\交通流预测研究\other_models\数据集\PeMS04\PeMS04.npz")
# print("原始数据形状: [节点数, 时间步长, 特征数]", pems_tensor.shape)
#
# # ----------------------
# # STSGCN模型定义
# # ----------------------
# class STSGCNLayer(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.linear = nn.Linear(input_dim, output_dim)
#
#     def forward(self, x, adj):
#         """x: [batch_size, total_nodes, input_dim]
#            adj: [total_nodes, total_nodes]
#         """
#         x = self.linear(x)
#         return torch.relu(torch.einsum('nm,bmc->bnc', adj, x))
#
# class STSGCN(nn.Module):
#     def __init__(self, num_nodes, input_dim, adj_ext, hidden_dim=64, out_dim=1):
#         super().__init__()
#         self.num_nodes = num_nodes
#         self.total_nodes = adj_ext.shape[0]  # num_nodes * window_size
#         self.window_size = self.total_nodes // num_nodes
#
#         self.stsgcn1 = STSGCNLayer(input_dim, hidden_dim)
#         self.stsgcn2 = STSGCNLayer(hidden_dim, hidden_dim)
#         self.stsgcn3 = STSGCNLayer(hidden_dim, hidden_dim)
#
#         self.final_fc = nn.Linear(self.total_nodes * hidden_dim, num_nodes * out_dim)
#         self.out_dim = out_dim
#
#     def forward(self, x):
#         # Input shape: [batch_size, num_nodes, window_size, input_dim]
#         batch_size = x.size(0)
#         x = x.view(batch_size, self.total_nodes, -1)  # [batch, total_nodes, input_dim]
#
#         # 时空图卷积
#         x = self.stsgcn1(x, adj_ext)
#         x = self.stsgcn2(x, adj_ext)
#         x = self.stsgcn3(x, adj_ext)
#
#         # 全连接输出
#         x = x.reshape(batch_size, -1)
#         x = self.final_fc(x)
#         return x.view(batch_size, self.num_nodes, self.out_dim)
#
# model = STSGCN(
#     num_nodes=pems_tensor.shape[0],
#     input_dim=pems_tensor.shape[-1],
#     adj_ext=adj_ext,
#     out_dim=pems_tensor.shape[-1]
# )
# print(model)
#
# # ----------------------
# # 时间窗口处理与数据划分（保持不变）
# # ----------------------
# def create_sequences(data, window_size=12):
#     num_nodes, total_timesteps, num_features = data.shape
#     seqs, labels = [], []
#     for i in range(total_timesteps - window_size):
#         seq = data[:, i:i+window_size, :]
#         label = data[:, i+window_size, :]
#         seqs.append(seq)
#         labels.append(label)
#     return torch.stack(seqs, 1), torch.stack(labels, 1)
#
# input_seq, target_seq = create_sequences(pems_tensor)
# print("输入序列形状:", input_seq.shape, "目标形状:", target_seq.shape)
#
# # 划分训练测试集
# total_samples = input_seq.shape[1]
# split_idx = int(total_samples * 0.8)
# train_input = input_seq[:, :split_idx, :, :]
# train_target = target_seq[:, :split_idx, :]
# test_input = input_seq[:, split_idx:, :, :]
# test_target = target_seq[:, split_idx:, :]
#
# # DataLoader
# train_dataset = TensorDataset(train_input.permute(1,0,2,3), train_target.permute(1,0,2))
# test_dataset = TensorDataset(test_input.permute(1,0,2,3), test_target.permute(1,0,2))
# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
#
# # ----------------------
# # 训练与评估
# # ----------------------
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# loss_fn = nn.MSELoss()
#
# print("开始训练STSGCN...")
# for epoch in range(3):
#     total_loss = 0
#     for inputs, targets in train_loader:
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = loss_fn(outputs, targets)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
#
# torch.save(model.state_dict(), "stsgcn_model.pth")
#
# def calculate_metrics(preds, labels):
#     epsilon = 1e-6
#     preds_flat = preds.view(-1)
#     labels_flat = labels.view(-1)
#
#     mae = torch.abs(preds_flat - labels_flat).mean().item()
#     rmse = torch.sqrt(torch.mean((preds_flat - labels_flat)**2)).item()
#     mape = torch.mean(torch.abs((labels_flat - preds_flat)/(labels_flat + epsilon)) * 100).item()
#
#     ss_res = torch.sum((labels_flat - preds_flat)**2)
#     ss_tot = torch.sum((labels_flat - torch.mean(labels_flat))**2)
#     r2 = (1 - (ss_res / (ss_tot + epsilon))).item()
#
#     return mae, rmse, mape, r2
#
# # 在测试评估部分修改打印语句
# mae, rmse, mape, r2 = calculate_metrics(all_preds, all_labels)
# print("测试结果：")
# print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.2f}% | R²: {r2:.4f}")


# -*- coding: utf-8 -*-
"""
STSGCN交通流预测模型
环境要求：Python 3.8+, PyTorch 1.10+, pandas, numpy
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ----------------------
# 步骤1：处理距离数据，构建邻接矩阵
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
        adj_matrix[i, j] = adj_matrix[j, i] = row['cost']

    print(f"总节点数: {num_nodes}, 示例距离(73->5): {adj_matrix[node_to_idx.get(73, 0), node_to_idx.get(5, 0)]:.1f}")
    return adj_matrix, node_to_idx


# 请修改为你的实际路径
adj_matrix, node_map = build_adjacency_matrix("E:\交通流预测研究\other_models\数据集\PeMS04\PeMS04.csv")


# ----------------------
# 步骤2：邻接矩阵归一化（空间部分）
# ----------------------
def normalize_adjacency(adj):
    sigma = 100.0
    adj_normalized = np.exp(-(adj ** 2) / (2 * sigma ** 2))
    np.fill_diagonal(adj_normalized, 1)
    return adj_normalized


adj_norm = normalize_adjacency(adj_matrix)


# ----------------------
# 步骤3：构造时空联合邻接矩阵
# ----------------------
def build_spatiotemporal_adj(space_adj, window_size):
    num_nodes = space_adj.shape[0]
    total = num_nodes * window_size
    adj_ext = np.zeros((total, total), dtype=np.float32)

    # 填充每个时间步的空间邻接
    for t in range(window_size):
        start = t * num_nodes
        end = start + num_nodes
        adj_ext[start:end, start:end] = space_adj

    # 填充时间邻接（相邻时间步同一节点）
    for t in range(window_size - 1):
        for i in range(num_nodes):
            idx1 = t * num_nodes + i
            idx2 = (t + 1) * num_nodes + i
            adj_ext[idx1, idx2] = adj_ext[idx2, idx1] = 1.0

    return adj_ext


window_size = 12  # 与后续create_sequences中参数一致
adj_ext = build_spatiotemporal_adj(adj_norm, window_size=window_size)
adj_ext = torch.FloatTensor(adj_ext)


# ----------------------
# 步骤4：加载并预处理数据
# ----------------------
def load_pems_data(npz_path):
    data = np.load(npz_path)
    flow_data = data['data']  # [T, N, F]
    # 转换为 [N, T, F] 并标准化数据（解决损失过大问题）
    flow_tensor = torch.FloatTensor(flow_data).permute(1, 0, 2)

    # 数据归一化到0-1范围
    mean = flow_tensor.mean()
    std = flow_tensor.std()
    flow_tensor = (flow_tensor - mean) / (std + 1e-8)

    return flow_tensor, mean, std


# 请修改为你的实际路径
pems_tensor, data_mean, data_std = load_pems_data("E:\交通流预测研究\other_models\数据集\PeMS04\PeMS04.npz")
print("原始数据形状: [节点数, 时间步长, 特征数]", pems_tensor.shape)


# ----------------------
# STSGCN模型定义
# ----------------------
class STSGCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, adj):
        """x: [batch_size, total_nodes, input_dim]
           adj: [total_nodes, total_nodes]
        """
        x = self.linear(x)
        return torch.relu(torch.einsum('nm,bmc->bnc', adj, x))


class STSGCN(nn.Module):
    def __init__(self, num_nodes, input_dim, adj_ext, hidden_dim=64, out_dim=1):
        super().__init__()
        self.num_nodes = num_nodes
        self.total_nodes = adj_ext.shape[0]  # num_nodes * window_size
        self.window_size = self.total_nodes // num_nodes
        self.adj_ext = adj_ext  # 存储邻接矩阵

        self.stsgcn1 = STSGCNLayer(input_dim, hidden_dim)
        self.stsgcn2 = STSGCNLayer(hidden_dim, hidden_dim)
        self.stsgcn3 = STSGCNLayer(hidden_dim, hidden_dim)

        self.final_fc = nn.Linear(self.total_nodes * hidden_dim, num_nodes * out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        # Input shape: [batch_size, num_nodes, window_size, input_dim]
        batch_size = x.size(0)
        x = x.view(batch_size, self.total_nodes, -1)  # [batch, total_nodes, input_dim]

        # 时空图卷积
        x = self.stsgcn1(x, self.adj_ext)
        x = self.stsgcn2(x, self.adj_ext)
        x = self.stsgcn3(x, self.adj_ext)

        # 全连接输出
        x = x.reshape(batch_size, -1)
        x = self.final_fc(x)
        return x.view(batch_size, self.num_nodes, self.out_dim)


model = STSGCN(
    num_nodes=pems_tensor.shape[0],
    input_dim=pems_tensor.shape[-1],
    adj_ext=adj_ext,
    out_dim=pems_tensor.shape[-1]
)
print(model)


# ----------------------
# 时间窗口处理与数据划分
# ----------------------
def create_sequences(data, window_size=12):
    num_nodes, total_timesteps, num_features = data.shape
    seqs, labels = [], []
    for i in range(total_timesteps - window_size):
        seq = data[:, i:i + window_size, :]
        label = data[:, i + window_size, :]
        seqs.append(seq)
        labels.append(label)
    return torch.stack(seqs, 1), torch.stack(labels, 1)


input_seq, target_seq = create_sequences(pems_tensor, window_size=window_size)
print("输入序列形状:", input_seq.shape, "目标形状:", target_seq.shape)

# 划分训练测试集
total_samples = input_seq.shape[1]
split_idx = int(total_samples * 0.8)
train_input = input_seq[:, :split_idx, :, :]
train_target = target_seq[:, :split_idx, :]
test_input = input_seq[:, split_idx:, :, :]
test_target = target_seq[:, split_idx:, :]

# DataLoader
train_dataset = TensorDataset(train_input.permute(1, 0, 2, 3), train_target.permute(1, 0, 2))
test_dataset = TensorDataset(test_input.permute(1, 0, 2, 3), test_target.permute(1, 0, 2))
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# ----------------------
# 训练与评估
# ----------------------
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

print("开始训练STSGCN...")
for epoch in range(3):
    model.train()  # 训练模式
    total_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.6f}")

torch.save(model.state_dict(), "stsgcn_model.pth")


def calculate_metrics(preds, labels):
    """计算评估指标，先将归一化数据还原"""
    # 还原数据到原始尺度
    preds = preds * data_std + data_mean
    labels = labels * data_std + data_mean

    epsilon = 1e-6
    preds_flat = preds.view(-1)
    labels_flat = labels.view(-1)

    mae = torch.abs(preds_flat - labels_flat).mean().item()
    rmse = torch.sqrt(torch.mean((preds_flat - labels_flat) ** 2)).item()
    mape = torch.mean(torch.abs((labels_flat - preds_flat) / (labels_flat + epsilon)) * 100).item()

    ss_res = torch.sum((labels_flat - preds_flat) ** 2)
    ss_tot = torch.sum((labels_flat - torch.mean(labels_flat)) ** 2)
    r2 = (1 - (ss_res / (ss_tot + epsilon))).item()

    return mae, rmse, mape, r2


# 测试评估（修复了all_preds未定义的问题）
model.eval()  # 评估模式
all_preds = []
all_labels = []

with torch.no_grad():  # 关闭梯度计算
    for inputs, targets in test_loader:
        outputs = model(inputs)
        all_preds.append(outputs)
        all_labels.append(targets)

# 合并所有批次的结果
all_preds = torch.cat(all_preds, dim=0)
all_labels = torch.cat(all_labels, dim=0)

# 计算并打印评估指标
mae, rmse, mape, r2 = calculate_metrics(all_preds, all_labels)
print("测试结果：")
print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.2f}% | R²: {r2:.4f}")
