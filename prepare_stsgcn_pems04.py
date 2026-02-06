"""
准备STSGCN的PeMS04数据
生成train.npz, val.npz, test.npz和adj_pems04.pkl
"""
import numpy as np
import pickle
import os

# 加载PeMS04数据
data_path = '/data_ssd/other_models/数据集/PeMS04/PeMS04.npz'
adj_path = '/data_ssd/other_models/数据集/PeMS04/PeMS04.csv'
output_dir = '/data_ssd/other_models/STSGCN_pytorch/data/PEMS04'

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 加载数据
data = np.load(data_path)
X = data['data']  # (16992, 307, 3)
print(f'原始数据形状: {X.shape}')

# 只使用flow特征 (第一个特征)
X = X[:, :, 0:1]  # (16992, 307, 1)
print(f'Flow数据形状: {X.shape}')

# 数据划分: 60% train, 20% val, 20% test
train_size = int(0.6 * X.shape[0])
val_size = int(0.2 * X.shape[0])
test_size = X.shape[0] - train_size - val_size

train_data = X[:train_size]
val_data = X[train_size:train_size+val_size]
test_data = X[train_size+val_size:]

print(f'训练集: {train_data.shape}')
print(f'验证集: {val_data.shape}')
print(f'测试集: {test_data.shape}')

# 生成时序样本
def generate_samples(data, seq_len=12, horizon=12):
    """
    生成输入输出序列
    输入: (T, N, F)
    输出: x-(samples, T, N, F), y-(samples, T, N, F)
    """
    num_samples = data.shape[0] - seq_len - horizon + 1
    x = np.zeros((num_samples, seq_len, data.shape[1], data.shape[2]))
    y = np.zeros((num_samples, horizon, data.shape[1], data.shape[2]))

    for i in range(num_samples):
        x[i] = data[i:i+seq_len]
        y[i] = data[i+seq_len:i+seq_len+horizon]

    return x, y

seq_len = 12
horizon = 12

# 生成训练集
x_train, y_train = generate_samples(train_data, seq_len, horizon)
print(f'训练样本: x={x_train.shape}, y={y_train.shape}')

# 生成验证集
x_val, y_val = generate_samples(val_data, seq_len, horizon)
print(f'验证样本: x={x_val.shape}, y={y_val.shape}')

# 生成测试集
x_test, y_test = generate_samples(test_data, seq_len, horizon)
print(f'测试样本: x={x_test.shape}, y={y_test.shape}')

# 保存npz文件
np.savez(os.path.join(output_dir, 'train.npz'), x=x_train, y=y_train)
np.savez(os.path.join(output_dir, 'val.npz'), x=x_val, y=y_val)
np.savez(os.path.join(output_dir, 'test.npz'), x=x_test, y=y_test)

print('\\n保存npz文件完成!')

# 构建邻接矩阵
print('\\n构建邻接矩阵...')
edges = np.loadtxt(adj_path, delimiter=',', skiprows=1)
num_nodes = 307
adj = np.zeros((num_nodes, num_nodes))

# 构建邻接矩阵（使用距离的倒数作为权重）
for edge in edges:
    i, j, dist = int(edge[0]), int(edge[1]), edge[2]
    if dist > 0:
        weight = 1.0 / dist
        adj[i, j] = weight
        adj[j, i] = weight  # 无向图

print(f'邻接矩阵形状: {adj.shape}')
print(f'边数: {len(edges)}, 非零元素: {np.count_nonzero(adj)}')

# 保存邻接矩阵为pkl文件
adj_output = os.path.join(output_dir, 'adj_pems04.pkl')
with open(adj_output, 'wb') as f:
    pickle.dump(adj, f)

print(f'\\n邻接矩阵已保存到: {adj_output}')

# 验证生成的数据
print('\\n验证数据:')
for category in ['train', 'val', 'test']:
    cat_data = np.load(os.path.join(output_dir, category + '.npz'))
    print(f'{category}: x={cat_data["x"].shape}, y={cat_data["y"].shape}')

print('\\nPeMS04数据准备完成!')
print(f'输出目录: {output_dir}')
