# -*- coding: utf-8 -*-
"""
新增R²计算
环境要求：Python 3.8+, PyTorch 1.10+, pandas, numpy
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ----------------------
# 步骤1：处理 distance.csv（保留但后续未使用）
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

    print(f"总节点数: {num_nodes}, 示例距离(73->5): {adj_matrix[node_to_idx[73], node_to_idx[5]]:.1f}")
    return adj_matrix, node_to_idx


adj_matrix, node_map = build_adjacency_matrix("E:\交通流预测研究\other_models\数据集\PeMS04\PeMS04.csv")


# ----------------------
# 步骤2：邻接矩阵归一化（保留但后续未使用）
# ----------------------
def normalize_adjacency(adj_matrix, sigma=100.0):
    adj_normalized = np.exp(-(adj_matrix ** 2) / (2 * sigma ** 2))
    np.fill_diagonal(adj_normalized, 1)
    return adj_normalized


adj_norm = normalize_adjacency(adj_matrix)


# ----------------------
# 步骤3：加载数据（保持不变）
# ----------------------
def load_pems_data(npz_path):
    data = np.load(npz_path)
    flow_data = data['data']  # [T, N, F]
    return torch.FloatTensor(flow_data).permute(1, 0, 2)  # [N, T, F]


pems_tensor = load_pems_data("E:\交通流预测研究\other_models\数据集\PeMS04\PeMS04.npz")
print("原始数据形状: [节点数, 时间步长, 特征数]", pems_tensor.shape)


# ----------------------
# 步骤4：定义LSTM模型（核心变更点）
# ----------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=2  # 使用2层LSTM增强时序建模能力
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 输入形状: [batch_size, num_nodes, window_size, feature]
        batch_size, num_nodes, window_size, feat_size = x.shape

        # 合并batch和nodes维度以并行处理
        x = x.view(-1, window_size, feat_size)  # [batch_size*num_nodes, window, feat]

        # LSTM处理时间序列
        out, _ = self.lstm(x)  # out形状: [batch*nodes, window, hidden]
        last_output = out[:, -1, :]  # 取最后时间步输出 [batch*nodes, hidden]

        # 投影到输出维度
        prediction = self.linear(last_output)  # [batch*nodes, output_feat]

        # 恢复原始维度
        return prediction.view(batch_size, num_nodes, -1)  # [batch, nodes, feat]


model = LSTMModel(
    input_size=pems_tensor.shape[-1],  # 输入特征维度
    hidden_size=64,  # LSTM隐藏单元数
    output_size=pems_tensor.shape[-1]  # 输出同输入特征维度
)
print(model)


# ----------------------
# 步骤5：时间窗口处理（保持不变）
# ----------------------
def create_sequences(data, window_size=12):
    num_nodes, total_timesteps, num_features = data.shape
    sequences = []
    targets = []

    for i in range(total_timesteps - window_size):
        seq = data[:, i:i + window_size, :]  # [N, window, F]
        label = data[:, i + window_size, :]  # [N, F]
        sequences.append(seq)
        targets.append(label)

    return torch.stack(sequences, dim=1), torch.stack(targets, dim=1)


input_seq, target_seq = create_sequences(pems_tensor, window_size=12)
print("输入序列形状:", input_seq.shape, "目标形状:", target_seq.shape)

# ----------------------
# 步骤6：数据划分与加载（保持不变）
# ----------------------
total_samples = input_seq.shape[1]
split_idx = int(total_samples * 0.8)

train_input = input_seq[:, :split_idx, :, :]
train_target = target_seq[:, :split_idx, :]

test_input = input_seq[:, split_idx:, :, :]
test_target = target_seq[:, split_idx:, :]

train_dataset = TensorDataset(
    train_input.permute(1, 0, 2, 3),
    train_target.permute(1, 0, 2)
)
test_dataset = TensorDataset(
    test_input.permute(1, 0, 2, 3),
    test_target.permute(1, 0, 2)
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# ----------------------
# 训练循环（保持结构，修改超参数）
# ----------------------
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(15):  # 增加训练轮数以适应LSTM特性
    total_loss = 0
    model.train()
    for batch_inputs, batch_targets in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = loss_fn(outputs, batch_targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

torch.save(model.state_dict(), "lstm_model_final.pth")


# ----------------------
# 步骤7：评估指标（保持不变）
# ----------------------

def calculate_metrics(preds, labels):
    epsilon = 1e-6
    mae = torch.abs(preds - labels).mean().item()
    rmse = torch.sqrt(torch.mean((preds - labels) ** 2)).item()
    mape = torch.mean(torch.abs((labels - preds) / (labels + epsilon)) * 100).item()

    # 新增R²计算
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

mae, rmse, mape, r2 = calculate_metrics(all_preds, all_labels)  # 接收四个返回值

print("\n最终评估结果（15分钟预测）：")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"R² Score: {r2:.4f}")  