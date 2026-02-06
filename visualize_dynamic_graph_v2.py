"""
动态图邻接矩阵可视化 v2
生成高峰期vs非高峰期的邻接矩阵热力图

改进版：
1. 使用二值化的连接矩阵（有/无连接）
2. 基于交通流相关性生成动态图
3. 更清晰的可视化
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from scipy.stats import pearsonr
import os

# 设置绘图风格
plt.style.use('seaborn-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.dpi'] = 150


def load_pems04_data():
    """加载PeMS04数据"""
    data = np.load('/data_ssd/other_models/数据集/PeMS04/PeMS04.npz')
    traffic_data = data['data'][:, :, 0]  # [T, N] 交通流量
    
    # 加载邻接矩阵
    edges = np.loadtxt('/data_ssd/other_models/数据集/PeMS04/PeMS04.csv', delimiter=',', skiprows=1)
    num_nodes = 307
    
    # 构建二值邻接矩阵（有连接=1，无连接=0）
    adj_binary = np.zeros((num_nodes, num_nodes))
    # 构建带权重的邻接矩阵
    adj_weighted = np.zeros((num_nodes, num_nodes))
    
    for edge in edges:
        i, j, dist = int(edge[0]), int(edge[1]), edge[2]
        adj_binary[i, j] = 1
        adj_binary[j, i] = 1
        if dist > 0:
            adj_weighted[i, j] = 1.0 / dist
            adj_weighted[j, i] = 1.0 / dist
    
    return traffic_data, adj_binary, adj_weighted, num_nodes


def compute_correlation_matrix(traffic_data, start_idx, end_idx, num_nodes):
    """计算节点间的交通流相关性矩阵"""
    data_slice = traffic_data[start_idx:end_idx, :]  # [T_slice, N]
    
    corr_matrix = np.zeros((num_nodes, num_nodes))
    
    for i in range(num_nodes):
        for j in range(i, num_nodes):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                # 计算皮尔逊相关系数
                corr, _ = pearsonr(data_slice[:, i], data_slice[:, j])
                if np.isnan(corr):
                    corr = 0
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
    
    return corr_matrix


def get_time_indices(time_period, samples_per_day=288):
    """获取特定时段的索引范围"""
    # 每5分钟一个样本，一天288个样本
    if time_period == 'off_peak':
        # 凌晨 2:00-4:00
        return 24, 48
    elif time_period == 'morning_peak':
        # 早高峰 7:00-9:00
        return 84, 108
    elif time_period == 'evening_peak':
        # 晚高峰 17:00-19:00
        return 204, 228
    elif time_period == 'midday':
        # 中午 12:00-14:00
        return 144, 168
    else:
        return 0, samples_per_day


def plot_adjacency_matrix(matrix, title, filename, vmin=None, vmax=None, cmap='Blues', subset=50):
    """绘制邻接矩阵热力图"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 只显示部分节点
    matrix_subset = matrix[:subset, :subset]
    
    im = ax.imshow(matrix_subset, cmap=cmap, aspect='equal', vmin=vmin, vmax=vmax)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Node Index', fontsize=12)
    ax.set_ylabel('Node Index', fontsize=12)
    
    # 设置刻度
    ticks = np.arange(0, subset, 10)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  保存: {filename}")


def plot_comparison_figure(static_adj, peak_corr, offpeak_corr, filename, subset=50):
    """绘制三合一对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    matrices = [
        (static_adj[:subset, :subset], 'Static Graph\n(Road Network)', 'Greys'),
        (peak_corr[:subset, :subset], 'Dynamic Graph\n(Morning Peak 7-9 AM)', 'Reds'),
        (offpeak_corr[:subset, :subset], 'Dynamic Graph\n(Off-Peak 2-4 AM)', 'Blues')
    ]
    
    for ax, (matrix, title, cmap) in zip(axes, matrices):
        if 'Static' in title:
            im = ax.imshow(matrix, cmap=cmap, aspect='equal', vmin=0, vmax=1)
        else:
            im = ax.imshow(matrix, cmap=cmap, aspect='equal', vmin=-1, vmax=1)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Node Index', fontsize=10)
        ax.set_ylabel('Node Index', fontsize=10)
        
        ticks = np.arange(0, subset, 10)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.tick_params(labelsize=9)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=9)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  保存: {filename}")


def plot_difference_figure(peak_corr, offpeak_corr, filename, subset=50):
    """绘制差异热力图"""
    diff = peak_corr[:subset, :subset] - offpeak_corr[:subset, :subset]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(diff, cmap='RdBu_r', aspect='equal', vmin=-0.5, vmax=0.5)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation Difference', fontsize=12)
    
    ax.set_title('Dynamic Graph Difference\n(Morning Peak - Off-Peak)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Node Index', fontsize=12)
    ax.set_ylabel('Node Index', fontsize=12)
    
    ticks = np.arange(0, subset, 10)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  保存: {filename}")


def plot_correlation_distribution(peak_corr, offpeak_corr, filename):
    """绘制相关性分布对比"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 获取上三角元素（不包括对角线）
    mask = np.triu(np.ones_like(peak_corr, dtype=bool), k=1)
    peak_values = peak_corr[mask]
    offpeak_values = offpeak_corr[mask]
    
    # 直方图
    axes[0].hist(peak_values, bins=50, alpha=0.6, label='Morning Peak (7-9 AM)', 
                 color='#E53935', edgecolor='white')
    axes[0].hist(offpeak_values, bins=50, alpha=0.6, label='Off-Peak (2-4 AM)', 
                 color='#1E88E5', edgecolor='white')
    axes[0].set_xlabel('Correlation Coefficient', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Node Correlations', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].axvline(x=peak_values.mean(), color='#E53935', linestyle='--', linewidth=2, alpha=0.8)
    axes[0].axvline(x=offpeak_values.mean(), color='#1E88E5', linestyle='--', linewidth=2, alpha=0.8)
    
    # 箱线图
    bp = axes[1].boxplot([peak_values, offpeak_values], 
                         labels=['Morning Peak\n(7-9 AM)', 'Off-Peak\n(2-4 AM)'],
                         patch_artist=True)
    bp['boxes'][0].set_facecolor('#E53935')
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor('#1E88E5')
    bp['boxes'][1].set_alpha(0.6)
    
    axes[1].set_ylabel('Correlation Coefficient', fontsize=12)
    axes[1].set_title('Correlation Statistics', fontsize=12, fontweight='bold')
    
    # 添加统计信息
    peak_mean = peak_values.mean()
    offpeak_mean = offpeak_values.mean()
    axes[1].text(0.95, 0.95, f'Peak Mean: {peak_mean:.3f}\nOff-Peak Mean: {offpeak_mean:.3f}',
                 transform=axes[1].transAxes, fontsize=10, verticalalignment='top',
                 horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  保存: {filename}")


def plot_fused_graph(static_adj, dynamic_corr, alpha, title, filename, subset=50):
    """绘制融合后的图"""
    # 将相关性转换为正值权重
    dynamic_weight = (dynamic_corr + 1) / 2  # 映射到[0,1]
    
    # 融合
    fused = alpha * static_adj + (1 - alpha) * dynamic_weight
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(fused[:subset, :subset], cmap='YlOrRd', aspect='equal', vmin=0, vmax=1)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Edge Weight', fontsize=12)
    
    ax.set_title(f'{title}\n(α={alpha:.1f})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Node Index', fontsize=12)
    ax.set_ylabel('Node Index', fontsize=12)
    
    ticks = np.arange(0, subset, 10)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  保存: {filename}")


def main():
    print("=" * 60)
    print("动态图邻接矩阵可视化 v2")
    print("=" * 60)
    
    output_dir = '/data_ssd/other_models/IEEE/figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    print("\n1. 加载PeMS04数据...")
    traffic_data, adj_binary, adj_weighted, num_nodes = load_pems04_data()
    print(f"   节点数: {num_nodes}")
    print(f"   时间步: {traffic_data.shape[0]}")
    print(f"   静态图边数: {int(adj_binary.sum() / 2)}")
    
    # 计算不同时段的相关性矩阵
    print("\n2. 计算交通流相关性矩阵（使用多天数据平均）...")
    
    samples_per_day = 288
    num_days = min(traffic_data.shape[0] // samples_per_day, 30)  # 最多用30天
    print(f"   使用 {num_days} 天的数据")
    
    def compute_avg_correlation(time_period):
        """计算多天平均相关性"""
        start_offset, end_offset = get_time_indices(time_period)
        all_corrs = []
        
        for day in range(num_days):
            day_start = day * samples_per_day
            start_idx = day_start + start_offset
            end_idx = day_start + end_offset
            
            if end_idx < traffic_data.shape[0]:
                corr = compute_correlation_matrix(traffic_data, start_idx, end_idx, num_nodes)
                all_corrs.append(corr)
        
        return np.mean(all_corrs, axis=0)
    
    # 早高峰
    print("   计算早高峰相关性...")
    peak_corr = compute_avg_correlation('morning_peak')
    
    # 非高峰
    print("   计算非高峰相关性...")
    offpeak_corr = compute_avg_correlation('off_peak')
    
    # 晚高峰
    print("   计算晚高峰相关性...")
    evening_corr = compute_avg_correlation('evening_peak')
    
    # 生成图片
    print("\n3. 生成可视化图片...")
    
    # 静态图
    plot_adjacency_matrix(adj_binary, 'Static Adjacency Matrix (Road Network)',
                          f'{output_dir}/static_adj_binary.png', vmin=0, vmax=1, cmap='Greys')
    
    # 动态图 - 早高峰
    plot_adjacency_matrix(peak_corr, 'Dynamic Graph - Morning Peak (7-9 AM)',
                          f'{output_dir}/dynamic_morning_peak.png', vmin=-1, vmax=1, cmap='RdYlBu_r')
    
    # 动态图 - 非高峰
    plot_adjacency_matrix(offpeak_corr, 'Dynamic Graph - Off-Peak (2-4 AM)',
                          f'{output_dir}/dynamic_off_peak.png', vmin=-1, vmax=1, cmap='RdYlBu_r')
    
    # 三合一对比图
    plot_comparison_figure(adj_binary, peak_corr, offpeak_corr,
                           f'{output_dir}/graph_comparison.png')
    
    # 差异热力图
    plot_difference_figure(peak_corr, offpeak_corr,
                           f'{output_dir}/correlation_difference.png')
    
    # 相关性分布
    plot_correlation_distribution(peak_corr, offpeak_corr,
                                  f'{output_dir}/correlation_distribution.png')
    
    # 融合图示例
    plot_fused_graph(adj_binary, peak_corr, alpha=0.3, title='Fused Graph (Morning Peak)',
                     filename=f'{output_dir}/fused_graph_peak.png')
    
    # 统计信息
    print("\n4. 统计信息:")
    mask = np.triu(np.ones((num_nodes, num_nodes), dtype=bool), k=1)
    print(f"   早高峰平均相关性: {peak_corr[mask].mean():.4f}")
    print(f"   非高峰平均相关性: {offpeak_corr[mask].mean():.4f}")
    print(f"   晚高峰平均相关性: {evening_corr[mask].mean():.4f}")
    print(f"   高峰vs非高峰差异: {(peak_corr[mask].mean() - offpeak_corr[mask].mean()):.4f}")
    
    print("\n" + "=" * 60)
    print(f"所有图片已保存到: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
