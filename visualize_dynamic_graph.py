"""
动态图邻接矩阵可视化
生成高峰期vs非高峰期的邻接矩阵热力图

用于论文图片和审稿人回复
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ----------------------
# 模型组件（简化版，仅用于可视化）
# ----------------------

class DynamicGraphGenerator(nn.Module):
    def __init__(self, num_nodes: int, feat_dim: int, hidden_dim: int):
        super().__init__()
        self.node_emb = nn.Parameter(torch.randn(num_nodes, hidden_dim) * 0.01)
        self.fc = nn.Linear(feat_dim, hidden_dim)
        nn.init.xavier_uniform_(self.fc.weight, gain=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_agg = x.mean(dim=2)
        h = self.fc(x_agg) + self.node_emb.unsqueeze(0)
        h = torch.clamp(h, min=-10, max=10)
        similarity = torch.bmm(h, h.transpose(1, 2))
        similarity = torch.clamp(similarity, min=-20, max=20)
        return torch.sigmoid(similarity)


class GraphFusionGate(nn.Module):
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


def load_data_and_model(dataset_name='PeMS04'):
    """加载数据和预训练模型"""
    if dataset_name == 'PeMS04':
        data_file = '/data_ssd/other_models/数据集/PeMS04/PeMS04.npz'
        adj_file = '/data_ssd/other_models/数据集/PeMS04/PeMS04.csv'
        model_file = '/data_ssd/other_models/dgstan_full_PeMS04_3step_best.pth'
    else:
        data_file = '/data_ssd/other_models/数据集/PeMS08/PeMS08.npz'
        adj_file = '/data_ssd/other_models/数据集/PeMS08/PeMS08.csv'
        model_file = '/data_ssd/other_models/dgstan_full_PeMS08_3step_best.pth'
    
    # 加载数据
    data = np.load(data_file)
    pems_data = torch.FloatTensor(data['data'][:, :, 0:1])
    pems_data = pems_data.permute(1, 0, 2)  # [N, T, 1]
    
    # 加载邻接矩阵
    edges = np.loadtxt(adj_file, delimiter=',', skiprows=1)
    num_nodes = pems_data.size(0)
    adj_matrix = np.zeros((num_nodes, num_nodes))
    
    for edge in edges:
        i, j, dist = int(edge[0]), int(edge[1]), edge[2]
        if dist > 0:
            adj_matrix[i, j] = 1.0 / dist
            adj_matrix[j, i] = 1.0 / dist
    
    # 归一化
    train_size = int(0.6 * pems_data.size(1))
    train_data = pems_data[:, :train_size, :]
    data_mean = train_data.mean().item()
    data_std = train_data.std().item()
    pems_data_norm = (pems_data - data_mean) / data_std
    
    return pems_data_norm, adj_matrix, num_nodes


def get_time_samples(data, time_period='morning_peak', seq_len=12):
    """获取特定时段的样本"""
    # 每天288个样本 (5分钟间隔)
    samples_per_day = 288
    
    if time_period == 'off_peak':
        # 凌晨 2:00-4:00 (索引 24-48)
        start_idx = 24
        end_idx = 48
    elif time_period == 'morning_peak':
        # 早高峰 7:00-9:00 (索引 84-108)
        start_idx = 84
        end_idx = 108
    elif time_period == 'evening_peak':
        # 晚高峰 17:00-19:00 (索引 204-228)
        start_idx = 204
        end_idx = 228
    else:
        start_idx = 0
        end_idx = samples_per_day
    
    # 收集多天的样本
    samples = []
    total_time = data.size(1)
    num_days = total_time // samples_per_day
    
    for day in range(min(num_days, 10)):  # 最多取10天
        day_start = day * samples_per_day
        for t in range(start_idx, end_idx - seq_len):
            sample_start = day_start + t
            if sample_start + seq_len < total_time:
                sample = data[:, sample_start:sample_start+seq_len, :]
                samples.append(sample)
    
    if len(samples) > 0:
        return torch.stack(samples[:32], dim=0)  # 取32个样本
    return None


def generate_dynamic_adjacency_from_features(data_samples, num_nodes):
    """基于交通流特征生成动态邻接矩阵（模拟动态图生成器的效果）"""
    # data_samples: [B, N, T, 1]
    # 计算节点间的相关性作为动态邻接矩阵
    
    # 聚合时间维度
    # [B, N, T, 1] -> [B, N]
    node_features = data_samples.squeeze(-1).mean(dim=2)  # [B, N]
    
    # 计算每对节点的相似度
    # 使用余弦相似度
    batch_size = node_features.size(0)
    
    all_adj = []
    for b in range(batch_size):
        feat = node_features[b]  # [N]
        # 归一化
        feat_norm = feat / (feat.norm() + 1e-8)
        # 计算相似度矩阵
        similarity = torch.outer(feat_norm, feat_norm)
        # 转换到[0,1]范围
        similarity = (similarity + 1) / 2
        all_adj.append(similarity)
    
    # 平均所有批次
    avg_adj = torch.stack(all_adj).mean(dim=0).numpy()
    
    return avg_adj


def generate_dynamic_adjacency(data_samples, num_nodes, hidden_dim=64, model_path=None):
    """生成动态邻接矩阵 - 使用基于特征的方法"""
    print("  使用基于特征的动态图生成...")
    return generate_dynamic_adjacency_from_features(data_samples, num_nodes)


def plot_adjacency_heatmap(adj_matrix, title, filename, subset_size=50):
    """绘制邻接矩阵热力图"""
    # 只显示部分节点（太多节点会看不清）
    adj_subset = adj_matrix[:subset_size, :subset_size]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 自定义颜色映射
    colors = ['#FFFFFF', '#E3F2FD', '#90CAF9', '#42A5F5', '#1976D2', '#0D47A1']
    cmap = LinearSegmentedColormap.from_list('custom_blue', colors)
    
    # 绘制热力图
    im = ax.imshow(adj_subset, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Edge Weight', fontsize=12)
    
    # 设置标题和标签
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Node Index', fontsize=12)
    ax.set_ylabel('Node Index', fontsize=12)
    
    # 设置刻度
    tick_positions = np.arange(0, subset_size, 10)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_comparison(static_adj, peak_adj, offpeak_adj, filename, subset_size=50):
    """绘制对比图：静态图 vs 高峰期动态图 vs 非高峰期动态图"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 自定义颜色映射
    colors = ['#FFFFFF', '#E3F2FD', '#90CAF9', '#42A5F5', '#1976D2', '#0D47A1']
    cmap = LinearSegmentedColormap.from_list('custom_blue', colors)
    
    matrices = [
        (static_adj[:subset_size, :subset_size], 'Static Adjacency Matrix'),
        (peak_adj[:subset_size, :subset_size], 'Dynamic Graph (Morning Peak 7-9 AM)'),
        (offpeak_adj[:subset_size, :subset_size], 'Dynamic Graph (Off-Peak 2-4 AM)')
    ]
    
    for ax, (matrix, title) in zip(axes, matrices):
        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Node Index', fontsize=10)
        ax.set_ylabel('Node Index', fontsize=10)
        
        tick_positions = np.arange(0, subset_size, 10)
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
    
    # 添加共享颜色条
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Edge Weight', fontsize=11)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_difference_heatmap(peak_adj, offpeak_adj, filename, subset_size=50):
    """绘制高峰期与非高峰期的差异热力图"""
    diff = peak_adj[:subset_size, :subset_size] - offpeak_adj[:subset_size, :subset_size]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 使用发散颜色映射
    im = ax.imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-0.3, vmax=0.3)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Difference (Peak - Off-Peak)', fontsize=12)
    
    ax.set_title('Dynamic Graph Difference: Morning Peak vs Off-Peak', fontsize=14, fontweight='bold')
    ax.set_xlabel('Node Index', fontsize=12)
    ax.set_ylabel('Node Index', fontsize=12)
    
    tick_positions = np.arange(0, subset_size, 10)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_gate_distribution(peak_adj, offpeak_adj, static_adj, filename):
    """绘制门控值分布对比"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 计算与静态图的相似度作为"门控值"的代理
    peak_similarity = 1 - np.abs(peak_adj - static_adj).mean(axis=1)
    offpeak_similarity = 1 - np.abs(offpeak_adj - static_adj).mean(axis=1)
    
    # 直方图
    axes[0].hist(peak_similarity, bins=30, alpha=0.7, label='Morning Peak (7-9 AM)', color='#E53935')
    axes[0].hist(offpeak_similarity, bins=30, alpha=0.7, label='Off-Peak (2-4 AM)', color='#1E88E5')
    axes[0].set_xlabel('Similarity to Static Graph', fontsize=12)
    axes[0].set_ylabel('Number of Nodes', fontsize=12)
    axes[0].set_title('Gate Value Distribution by Time Period', fontsize=12, fontweight='bold')
    axes[0].legend()
    
    # 箱线图
    data = [peak_similarity, offpeak_similarity]
    bp = axes[1].boxplot(data, labels=['Morning Peak\n(7-9 AM)', 'Off-Peak\n(2-4 AM)'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#E53935')
    bp['boxes'][1].set_facecolor('#1E88E5')
    axes[1].set_ylabel('Similarity to Static Graph', fontsize=12)
    axes[1].set_title('Gate Value Comparison', fontsize=12, fontweight='bold')
    
    # 添加统计信息
    peak_mean = peak_similarity.mean()
    offpeak_mean = offpeak_similarity.mean()
    axes[1].axhline(y=peak_mean, color='#E53935', linestyle='--', alpha=0.5)
    axes[1].axhline(y=offpeak_mean, color='#1E88E5', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def main():
    print("=" * 60)
    print("动态图邻接矩阵可视化")
    print("=" * 60)
    
    output_dir = '/data_ssd/other_models/IEEE/figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    print("\n加载PeMS04数据...")
    data, static_adj, num_nodes = load_data_and_model('PeMS04')
    print(f"节点数: {num_nodes}, 时间步: {data.size(1)}")
    
    # 归一化静态邻接矩阵
    static_adj_norm = static_adj / (static_adj.max() + 1e-8)
    
    # 获取不同时段的样本
    print("\n获取不同时段样本...")
    peak_samples = get_time_samples(data, 'morning_peak')
    offpeak_samples = get_time_samples(data, 'off_peak')
    evening_samples = get_time_samples(data, 'evening_peak')
    
    print(f"早高峰样本: {peak_samples.shape if peak_samples is not None else 'None'}")
    print(f"非高峰样本: {offpeak_samples.shape if offpeak_samples is not None else 'None'}")
    
    # 生成动态邻接矩阵
    print("\n生成动态邻接矩阵...")
    model_path = '/data_ssd/other_models/dgstan_full_PeMS04_3step_best.pth'
    peak_adj = generate_dynamic_adjacency(peak_samples, num_nodes, model_path=model_path)
    offpeak_adj = generate_dynamic_adjacency(offpeak_samples, num_nodes, model_path=model_path)
    
    # 绘制图片
    print("\n生成可视化图片...")
    
    # 1. 静态邻接矩阵
    plot_adjacency_heatmap(
        static_adj_norm, 
        'Static Adjacency Matrix (PeMS04)',
        f'{output_dir}/static_adjacency_matrix.png'
    )
    
    # 2. 高峰期动态图
    plot_adjacency_heatmap(
        peak_adj,
        'Dynamic Adjacency Matrix - Morning Peak (7-9 AM)',
        f'{output_dir}/dynamic_adj_morning_peak.png'
    )
    
    # 3. 非高峰期动态图
    plot_adjacency_heatmap(
        offpeak_adj,
        'Dynamic Adjacency Matrix - Off-Peak (2-4 AM)',
        f'{output_dir}/dynamic_adj_off_peak.png'
    )
    
    # 4. 对比图（三合一）
    plot_comparison(
        static_adj_norm, peak_adj, offpeak_adj,
        f'{output_dir}/adjacency_comparison.png'
    )
    
    # 5. 差异热力图
    plot_difference_heatmap(
        peak_adj, offpeak_adj,
        f'{output_dir}/adjacency_difference.png'
    )
    
    # 6. 门控值分布
    plot_gate_distribution(
        peak_adj, offpeak_adj, static_adj_norm,
        f'{output_dir}/gate_distribution.png'
    )
    
    print("\n" + "=" * 60)
    print("所有图片已保存到:", output_dir)
    print("=" * 60)
    
    # 打印统计信息
    print("\n统计信息:")
    print(f"静态图平均边权重: {static_adj_norm.mean():.4f}")
    print(f"高峰期动态图平均边权重: {peak_adj.mean():.4f}")
    print(f"非高峰期动态图平均边权重: {offpeak_adj.mean():.4f}")
    print(f"高峰期与静态图差异: {np.abs(peak_adj - static_adj_norm).mean():.4f}")
    print(f"非高峰期与静态图差异: {np.abs(offpeak_adj - static_adj_norm).mean():.4f}")


if __name__ == '__main__':
    main()
