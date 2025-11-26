import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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
    return adj_matrix, node_to_idx

def normalize_adjacency(adj_matrix, sigma=100.0):
    adj_normalized = np.exp(-(adj_matrix ** 2) / (2 * sigma ** 2))
    np.fill_diagonal(adj_normalized, 1)
    return adj_normalized

def load_pems_data(npz_path):
    data = np.load(npz_path)
    flow_data = data['data']
    return torch.FloatTensor(flow_data).permute(1, 0, 2)

class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.weights = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        nn.init.xavier_uniform_(self.weights)
        nn.init.zeros_(self.bias)

    def forward(self, adj, x):
        x = torch.matmul(x, self.weights)  # [B*T, N, output_dim]
        x = torch.matmul(adj, x)           # [B*T, N, output_dim]
        return x + self.bias

class ASTGCNBlock(nn.Module):
    def __init__(self, adj, input_dim, output_dim):
        super().__init__()
        self.gcn = GraphConv(input_dim, output_dim)
        self.temporal_conv = nn.Conv1d(output_dim, output_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        batch, N, T, input_dim = x.shape
        x_reshaped = x.permute(0, 2, 1, 3).reshape(batch*T, N, input_dim)  # [B*T, N, input_dim]
        
        x_gcn = self.gcn(adj, x_reshaped)    # [B*T, N, output_dim]
        x_gcn = x_gcn.view(batch, T, N, -1)  # [B, T, N, H]
        x_gcn = x_gcn.permute(0, 2, 1, 3)    # [B, N, T, H]
        
        x_temporal = x_gcn.permute(0, 1, 3, 2)  # [B, N, H, T]
        x_temporal = x_temporal.reshape(batch*N, -1, T)  # [B*N, H, T]
        
        x_temporal = self.temporal_conv(x_temporal)  # [B*N, H, T]
        x_temporal = x_temporal.view(batch, N, -1, T)  # [B, N, H, T]
        
        return self.relu(x_temporal.permute(0, 1, 3, 2))  # [B, N, T, H]

class ASTGCN(nn.Module):
    def __init__(self, num_nodes, input_dim, adj_matrix, hidden=64):
        super().__init__()
        self.register_buffer('adj', torch.FloatTensor(adj_matrix))
        self.block1 = ASTGCNBlock(self.adj, input_dim, hidden)
        self.block2 = ASTGCNBlock(self.adj, hidden, hidden)
        self.final = nn.Linear(hidden, input_dim)

    def forward(self, x):
        x = self.block1(x, self.adj)  # [B, N, T, H]
        x = self.block2(x, self.adj)  # [B, N, T, H]
        return self.final(x[:, :, -1, :])  # Predict last timestep

def create_sequences(data, window_size=12):
    num_nodes, num_timesteps, num_features = data.shape
    sequences = []
    targets = []
    for i in range(num_timesteps - window_size):
        seq = data[:, i:i+window_size, :]
        label = data[:, i+window_size, :]
        sequences.append(seq)
        targets.append(label)
    return torch.stack(sequences, dim=1), torch.stack(targets, dim=1)

# Main execution
if __name__ == "__main__":
    adj_matrix, node_map = build_adjacency_matrix("E:\交通流预测研究\other_models\数据集\PeMS04\PeMS04.csv")
    adj_norm = normalize_adjacency(adj_matrix)
    pems_tensor = load_pems_data("E:\交通流预测研究\other_models\数据集\PeMS04\PeMS04.npz")
    input_seq, target_seq = create_sequences(pems_tensor)

    # Data splitting and loader preparation
    num_nodes, num_sequences, window_size, num_features = input_seq.shape
    split = int(num_sequences * 0.8)
    
    train_input = input_seq[:, :split, :, :]
    train_target = target_seq[:, :split, :]
    test_input = input_seq[:, split:, :, :]
    test_target = target_seq[:, split:, :]

    # Correct dimension order for DataLoader: (Samples, Nodes, Time, Features)
    train_loader = DataLoader(TensorDataset(
        train_input.permute(1, 0, 2, 3),  # [num_train, N, T, F]
        train_target.permute(1, 0, 2)),    # [num_train, N, F]
        batch_size=8, shuffle=True)
    
    test_loader = DataLoader(TensorDataset(
        test_input.permute(1, 0, 2, 3),
        test_target.permute(1, 0, 2)),
        batch_size=8, shuffle=False)

    model = ASTGCN(num_nodes=pems_tensor.shape[0],
                   input_dim=pems_tensor.shape[-1],
                   adj_matrix=adj_norm,
                   hidden=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # Training loop
    for epoch in range(10):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)  # input: [B, N, T, F]
            loss = loss_fn(outputs.permute(1, 0, 2), targets.permute(1, 0, 2))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

       # 评估部分
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            all_preds.append(outputs)
            all_labels.append(targets)
        preds = torch.cat(all_preds, dim=0).permute(1, 0, 2)
        labels = torch.cat(all_labels, dim=0).permute(1, 0, 2)
        
        # 计算基础指标
        mae = torch.abs(preds - labels).mean().item()
        rmse = torch.sqrt(torch.mean((preds - labels)**2)).item()
        mape = torch.mean(torch.abs((labels - preds)/(labels + 1e-6)) * 100).item()
        
        # 新增R2计算
        ss_res = torch.sum((preds - labels)**2)
        ss_tot = torch.sum((labels - torch.mean(labels))**2)
        r2 = (1 - ss_res / ss_tot).item()
        
        print(f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nMAPE: {mape:.2f}%\nR²: {r2:.4f}")