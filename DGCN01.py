# -*- coding: utf-8 -*-
"""
DGCN 实现版本（基于原DG-STAN代码修改）
环境要求：Python 3.8+, PyTorch 1.10+, pandas, numpy
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ----------------------
# 步骤1：处理 pems.csv（保持不变）
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
# 步骤2：邻接矩阵归一化（保持不变）
# ----------------------
def normalize_adjacency(adj_matrix, sigma=100.0):
    adj_normalized = np.exp(-(adj_matrix ** 2) / (2 * sigma ** 2))
    np.fill_diagonal(adj_normalized, 1)
    return adj_normalized

adj_norm = normalize_adjacency(adj_matrix)

# ----------------------
# 步骤3：加载数据（维度修复）
# ----------------------
def load_pems_data(npz_path):
    data = np.load(npz_path)
    flow_data = data['data']  # [T, N, F]
    return torch.FloatTensor(flow_data).permute(1, 0, 2)  # [N, T, F]

pems_tensor = load_pems_data("E:\交通流预测研究\other_models\数据集\PeMS04\PeMS04.npz")
print("原始数据形状: [节点数, 时间步长, 特征数]", pems_tensor.shape)

# ----------------------
# 修改部分：DGCN模型核心组件
# ----------------------
class DiffusionConv(nn.Module):
    def __init__(self, input_dim, output_dim, adj_matrix, K=2):
        """
        DGCN扩散卷积层
        K: 扩散步数（双向各K步）
        """
        super().__init__()
        self.K = K
        self.num_nodes = adj_matrix.shape[0]
        
        # 构建正向反向转移矩阵
        adj_matrix = torch.FloatTensor(adj_matrix)
        adj_matrix += torch.eye(self.num_nodes)  # 添加自环
        
        # 随机游走归一化
        rowsum = adj_matrix.sum(1)
        rowsum[rowsum == 0] = 1  # 处理零度节点
        d_mat_inv = torch.diag(1.0 / rowsum)
        
        self.A_fw = torch.mm(d_mat_inv, adj_matrix).t()    # 正向转移
        self.A_bw = torch.mm(d_mat_inv, adj_matrix).t()    # 反向转移
        
        # 可训练参数
        self.weights = nn.Parameter(torch.FloatTensor(input_dim*(2*K+1), output_dim))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
    
    def forward(self, x):
        """
        输入形状： (batch_size*num_nodes, input_dim)
        输出形状： (batch_size*num_nodes, output_dim)
        """
        x = x.reshape(-1, self.num_nodes, x.size(-1))  # [batch_size, num_nodes, dim]
        
        supports = [x]  # 初始状态
        x_fw = x.clone()
        x_bw = x.clone()
        
        # 扩散过程
        for _ in range(self.K):
            x_fw = torch.einsum("nm,bmd->bnd", self.A_fw, x_fw)
            x_bw = torch.einsum("nm,bmd->bnd", self.A_bw, x_bw)
            supports.append(x_fw)
            supports.append(x_bw)
        
        # 融合所有扩散步特征
        h = torch.cat(supports, dim=2)                      # [batch, N, dim*(2K+1)]
        h = h.reshape(-1, h.size(2))                        # [batch*N, dim*(2K+1)]
        output = torch.mm(h, self.weights)                  # [batch*N, out_dim]
        return output

class DGCN(nn.Module):
    def __init__(self, num_nodes, input_dim, adj_matrix, hidden_dim=64, K=2):
        super().__init__()
        self.num_nodes = num_nodes
        
        # 图卷积层
        self.diffusion_conv = DiffusionConv(
            input_dim=input_dim,
            output_dim=hidden_dim,
            adj_matrix=adj_matrix,
            K=K
        )
        
        # 时间门控单元
        self.temporal_grus = nn.ModuleList([
            nn.GRU(hidden_dim, hidden_dim, batch_first=True)
            for _ in range(num_nodes)  # 每个节点独立的时间建模
        ])
        
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        """输入形状：[batch, N, window, features]"""
        batch_size, num_nodes, window_size, _ = x.shape
        
        # 空间特征提取
        x = x.permute(0, 2, 1, 3)                           # [batch, window, N, F]
        x = x.reshape(batch_size * window_size, num_nodes, -1) # [(batch*win), N, F]
        x = x.reshape(-1, x.size(-1))                       # [(batch*win*N), F]
        
        spatial_feat = self.diffusion_conv(x)              # [(batch*win*N), hidden]
        spatial_feat = spatial_feat.view(
            batch_size * window_size, num_nodes, -1
        ).permute(1, 0, 2)                                 # [N, (batch*win), hidden]

        # 时间特征提取（每个节点独立处理）
        temporal_feats = []
        for i in range(num_nodes):
            node_feat = spatial_feat[i]                    # [batch*win, hidden]
            node_feat = node_feat.view(batch_size, window_size, -1)
            out, _ = self.temporal_grus[i](node_feat)      # [batch, win, hidden]
            temporal_feats.append(out[:, -1, :])           # 取最后时间步
            
        feat = torch.stack(temporal_feats, dim=1)          # [batch, N, hidden]
        
        return self.output_layer(feat)

# ----------------------
# 下面的步骤与原代码保持一致（稍作模型参数调整）
# ----------------------
model = DGCN(
    num_nodes=pems_tensor.shape[0],
    input_dim=pems_tensor.shape[-1],
    adj_matrix=adj_norm,
    K=2
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
        seq = data[:, i:i+window_size, :]  
        label = data[:, i+window_size, :]  
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
print("训练集样本数:", len(train_dataset), "测试集样本数:", len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# ----------------------
# 训练循环（保持结构，微调日志输出）
# ----------------------
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(10):  # 适当增加训练轮数
    total_loss = 0
    for batch_inputs, batch_targets in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = loss_fn(outputs, batch_targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1:02d}, Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "dgcn_model_final.pth")


# ----------------------
# 评估阶段（添加R²计算）
# ----------------------
def calculate_metrics(preds, labels):
    epsilon = 1e-6
    mae = torch.abs(preds - labels).mean().item()
    rmse = torch.sqrt(torch.mean((preds - labels)**2)).item()
    mape = torch.mean(torch.abs((labels - preds)/(labels + epsilon)) * 100).item()
    
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

print("\n最终评估结果（基于DGCN的15分钟预测）：")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"R² Score: {r2:.4f}")  # 新增输出行
