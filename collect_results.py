#!/usr/bin/env python3
"""
收集STAEformer和STGCN的所有实验结果
"""
import re
import os
from pathlib import Path

def extract_staeformer_results(log_file):
    """从STAEformer日志中提取测试结果"""
    try:
        with open(log_file, 'r') as f:
            content = f.read()

        # 查找测试结果部分: All Steps RMSE = ..., MAE = ..., MAPE = ..., R² = ...
        pattern = r'All Steps RMSE = ([\d.]+), MAE = ([\d.]+), MAPE = ([\d.]+), R² = ([\d.]+)'
        match = re.search(pattern, content)

        if match:
            return {
                'RMSE': float(match.group(1)),
                'MAE': float(match.group(2)),
                'MAPE': float(match.group(3)),
                'R²': float(match.group(4))
            }
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
    return None

def extract_stgcn_results(log_file):
    """从STGCN日志中提取测试结果"""
    try:
        with open(log_file, 'r') as f:
            content = f.read()

        # 查找最终测试结果
        lines = content.split('\n')
        results = {}

        for i, line in enumerate(lines):
            if 'MAE:' in line and 'RMSE:' not in line:
                # MAE:    19.67
                match = re.search(r'MAE:\s*([\d.]+)', line)
                if match:
                    results['MAE'] = float(match.group(1))
            elif 'RMSE:' in line and i > 0:
                match = re.search(r'RMSE:\s*([\d.]+)', line)
                if match:
                    results['RMSE'] = float(match.group(1))
            elif 'MAPE:' in line:
                match = re.search(r'MAPE:\s*([\d.]+)%?', line)
                if match:
                    results['MAPE'] = float(match.group(1))
            elif 'R²:' in line or 'R2:' in line:
                match = re.search(r'R[²2]:\s*([\d.]+)', line)
                if match:
                    results['R²'] = float(match.group(1))

        if len(results) >= 3:  # 至少有MAE, RMSE, MAPE
            return results
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
    return None

def main():
    base_dir = Path('/data_ssd/other_models/baseline_models')

    # 定义实验配置
    experiments = [
        # STAEformer experiments
        ('STAEformer', 'PEMS04', '3step', 'STAEformer/staeformer_3step.log'),
        ('STAEformer', 'PEMS04', '6step', 'STAEformer/staeformer_6step.log'),
        ('STAEformer', 'PEMS04', '12step', 'STAEformer/staeformer_train.log'),  # 12step默认
        ('STAEformer', 'PEMS08', '3step', 'STAEformer/staeformer_pems08_3step.log'),
        ('STAEformer', 'PEMS08', '6step', 'STAEformer/staeformer_pems08_6step.log'),
        ('STAEformer', 'PEMS08', '12step', 'STAEformer/staeformer_pems08_12step.log'),
        ('STAEformer', 'METR-LA', '3step', 'STAEformer/staeformer_metrla_3step.log'),
        ('STAEformer', 'METR-LA', '6step', 'STAEformer/staeformer_metrla_6step.log'),
        ('STAEformer', 'METR-LA', '12step', 'STAEformer/staeformer_metrla_12step.log'),

        # STGCN experiments
        ('STGCN', 'PEMS04', '3step', 'STGCN/results_PEMS04/stgcn_3step.log'),
        ('STGCN', 'PEMS04', '6step', 'STGCN/results_PEMS04/stgcn_6step.log'),
        ('STGCN', 'PEMS04', '12step', 'STGCN/results_PEMS04/stgcn_12step.log'),
        ('STGCN', 'PEMS08', '3step', 'STGCN/stgcn_pems08_3step.log'),
        ('STGCN', 'PEMS08', '6step', 'STGCN/stgcn_pems08_6step.log'),
        ('STGCN', 'PEMS08', '12step', 'STGCN/stgcn_pems08.log'),
        ('STGCN', 'METR-LA', '3step', 'STGCN/stgcn_metrla_3step_fixed.log'),
        ('STGCN', 'METR-LA', '6step', 'STGCN/stgcn_metrla_6step_fixed.log'),
        ('STGCN', 'METR-LA', '12step', 'STGCN/stgcn_metrla_12step_fixed.log'),
    ]

    print("=" * 100)
    print("收集基准模型实验结果")
    print("=" * 100)
    print()

    # 按数据集和步长分组
    datasets = ['PEMS04', 'PEMS08', 'METR-LA']
    steps = ['3step', '6step', '12step']

    results_table = {}

    for model, dataset, step, log_path in experiments:
        full_path = base_dir / log_path

        if not full_path.exists():
            print(f"⚠️  文件不存在: {log_path}")
            continue

        if model == 'STAEformer':
            results = extract_staeformer_results(full_path)
        else:
            results = extract_stgcn_results(full_path)

        if results:
            key = f"{dataset}_{step}"
            if key not in results_table:
                results_table[key] = {}
            results_table[key][model] = results
            print(f"✅ {model:12s} | {dataset:8s} | {step:6s} | MAE={results.get('MAE', 0):.2f} | RMSE={results.get('RMSE', 0):.2f} | MAPE={results.get('MAPE', 0):.2f}% | R²={results.get('R²', 0):.4f}")
        else:
            print(f"❌ {model:12s} | {dataset:8s} | {step:6s} | 无法提取结果")

    print()
    print("=" * 100)
    print("详细对比表格")
    print("=" * 100)
    print()

    for dataset in datasets:
        print(f"\n{'='*100}")
        print(f"数据集: {dataset}")
        print(f"{'='*100}")
        print(f"{'步长':<10s} | {'模型':<12s} | {'MAE':<10s} | {'RMSE':<10s} | {'MAPE':<10s} | {'R²':<10s}")
        print("-" * 100)

        for step in steps:
            key = f"{dataset}_{step}"
            if key in results_table:
                for model in ['STAEformer', 'STGCN']:
                    if model in results_table[key]:
                        r = results_table[key][model]
                        print(f"{step:<10s} | {model:<12s} | {r.get('MAE', 0):<10.2f} | {r.get('RMSE', 0):<10.2f} | {r.get('MAPE', 0):<10.2f} | {r.get('R²', 0):<10.4f}")

                # 计算差异
                if 'STAEformer' in results_table[key] and 'STGCN' in results_table[key]:
                    stae = results_table[key]['STAEformer']
                    stgcn = results_table[key]['STGCN']

                    mae_diff = ((stae.get('MAE', 0) - stgcn.get('MAE', 0)) / stgcn.get('MAE', 1)) * 100
                    rmse_diff = ((stae.get('RMSE', 0) - stgcn.get('RMSE', 0)) / stgcn.get('RMSE', 1)) * 100

                    winner = "STAEformer" if mae_diff < 0 else "STGCN"
                    print(f"{'':10s} | 差异 ({winner}更好) | {mae_diff:+.2f}%    | {rmse_diff:+.2f}%    |")
                print("-" * 100)

    print()
    print("=" * 100)
    print("统计摘要")
    print("=" * 100)

    # 统计哪个模型更好
    stae_wins = 0
    stgcn_wins = 0

    for key in results_table:
        if 'STAEformer' in results_table[key] and 'STGCN' in results_table[key]:
            stae_mae = results_table[key]['STAEformer'].get('MAE', float('inf'))
            stgcn_mae = results_table[key]['STGCN'].get('MAE', float('inf'))

            if stae_mae < stgcn_mae:
                stae_wins += 1
            else:
                stgcn_wins += 1

    total = stae_wins + stgcn_wins
    print(f"STAEformer 更好: {stae_wins}/{total} ({stae_wins/total*100:.1f}%)")
    print(f"STGCN 更好: {stgcn_wins}/{total} ({stgcn_wins/total*100:.1f}%)")

if __name__ == '__main__':
    main()
