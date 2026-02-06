#!/usr/bin/env python3
"""
检查各个数据集中0值的比例
评估masked metrics的潜在影响
"""
import numpy as np
import os

def analyze_dataset(data_path, dataset_name):
    """分析数据集中的0值分布"""
    print(f"\n{'='*80}")
    print(f"数据集: {dataset_name}")
    print(f"{'='*80}")

    if not os.path.exists(data_path):
        print(f"❌ 文件不存在: {data_path}")
        return

    # 加载数据
    data = np.load(data_path)

    if 'data' in data:
        X = data['data']
    else:
        print(f"可用的keys: {list(data.keys())}")
        return

    print(f"数据形状: {X.shape}")  # (T, N, F)

    # 只分析flow特征（第一个特征）
    flow = X[:, :, 0]
    print(f"Flow特征形状: {flow.shape}")

    # 统计0值
    total_values = flow.size
    zero_values = np.sum(flow == 0)
    zero_ratio = (zero_values / total_values) * 100

    print(f"\n零值统计:")
    print(f"  总数值点: {total_values:,}")
    print(f"  零值数量: {zero_values:,}")
    print(f"  零值比例: {zero_ratio:.4f}%")

    # 统计接近0的值（小于阈值）
    thresholds = [1, 5, 10]
    print(f"\n低值统计:")
    for thresh in thresholds:
        low_values = np.sum(flow < thresh)
        low_ratio = (low_values / total_values) * 100
        print(f"  < {thresh:2d} 的值: {low_values:,} ({low_ratio:.4f}%)")

    # 统计描述
    print(f"\n描述统计:")
    print(f"  最小值: {np.min(flow):.2f}")
    print(f"  最大值: {np.max(flow):.2f}")
    print(f"  均值:   {np.mean(flow):.2f}")
    print(f"  中位数: {np.median(flow):.2f}")
    print(f"  标准差: {np.std(flow):.2f}")

    # 统计各个时间步的0值比例
    print(f"\n时间维度零值分布:")
    zero_per_timestep = np.sum(flow == 0, axis=1)
    print(f"  每个时间步平均零值节点数: {np.mean(zero_per_timestep):.2f}")
    print(f"  最多零值的时间步: {np.max(zero_per_timestep)} 个节点")
    print(f"  最少零值的时间步: {np.min(zero_per_timestep)} 个节点")

    # 统计各个节点的0值比例
    print(f"\n空间维度零值分布:")
    zero_per_node = np.sum(flow == 0, axis=0)
    print(f"  每个节点平均零值时间步: {np.mean(zero_per_node):.2f}")
    print(f"  最多零值的节点: {np.max(zero_per_node)} 个时间步")
    print(f"  最少零值的节点: {np.min(zero_per_node)} 个时间步")
    print(f"  完全无零值的节点数: {np.sum(zero_per_node == 0)}")

    # 评估masked metrics的潜在影响
    print(f"\n⚠️  Masked Metrics影响评估:")
    if zero_ratio > 10:
        print(f"  🔴 严重: {zero_ratio:.2f}% 的值会被忽略，结果可能严重偏差")
    elif zero_ratio > 5:
        print(f"  🟡 中等: {zero_ratio:.2f}% 的值会被忽略，结果可能有偏差")
    elif zero_ratio > 1:
        print(f"  🟢 轻微: {zero_ratio:.2f}% 的值会被忽略，影响较小")
    else:
        print(f"  ✅ 最小: {zero_ratio:.2f}% 的值会被忽略，影响可忽略")

    return {
        'dataset': dataset_name,
        'zero_ratio': zero_ratio,
        'mean': np.mean(flow),
        'std': np.std(flow),
        'min': np.min(flow),
        'max': np.max(flow)
    }


def main():
    print("=" * 80)
    print("数据集零值分析")
    print("=" * 80)

    datasets = [
        ('STGCN/data/PEMS04.npz', 'PEMS04'),
        ('STGCN/data/PEMS08.npz', 'PEMS08'),
        ('STAEformer/data/METRLA/data.npz', 'METR-LA'),
    ]

    results = []
    for data_path, name in datasets:
        full_path = os.path.join('/data_ssd/other_models/baseline_models', data_path)
        result = analyze_dataset(full_path, name)
        if result:
            results.append(result)

    # 汇总表格
    print(f"\n\n{'='*80}")
    print("汇总表格")
    print(f"{'='*80}")
    print(f"{'数据集':<12s} | {'零值比例':<12s} | {'均值':<10s} | {'标准差':<10s} | {'最小值':<10s} | {'最大值':<10s}")
    print("-" * 80)
    for r in results:
        print(f"{r['dataset']:<12s} | {r['zero_ratio']:>10.4f}%  | {r['mean']:>8.2f}  | {r['std']:>8.2f}  | {r['min']:>8.2f}  | {r['max']:>8.2f}")

    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)
    print("""
如果零值比例较高，STAEformer的masked metrics会显著低估真实误差，因为：
1. 零值或低流量时段通常更难预测
2. 忽略这些时段相当于"挑选"简单的预测任务
3. 这导致与STGCN（使用标准metrics）的不公平对比

建议：使用一致的评估指标（标准或masked）重新评估所有模型。
    """)


if __name__ == '__main__':
    main()
