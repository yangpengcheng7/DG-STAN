#!/usr/bin/env python
"""
评估当前正在训练的所有模型
"""
import os
import sys
import numpy as np
import torch

# ============================================================================
# 1. 评估 DCRNN PeMS08 12-step
# ============================================================================
def evaluate_dcrnn_pems08_12step():
    print("=" * 60)
    print("评估 DCRNN PeMS08 12-step")
    print("=" * 60)
    
    sys.path.insert(0, '/data_ssd/other_models/baseline_models/DCRNN_PyTorch')
    os.chdir('/data_ssd/other_models/baseline_models/DCRNN_PyTorch')
    
    import yaml
    from lib.utils import load_graph_data
    from model.pytorch.dcrnn_supervisor import DCRNNSupervisor
    
    # 加载配置
    config_file = 'data/model/dcrnn_pems08_12step.yaml'
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    # 找到最佳模型
    model_dir = 'models_pems08_12step'
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.tar')]
    if not model_files:
        print("没有找到模型文件")
        return None
    
    # 选择最新的模型
    model_files.sort(key=lambda x: int(x.replace('epo', '').replace('.tar', '')))
    best_model = model_files[-1]
    best_epoch = int(best_model.replace('epo', '').replace('.tar', ''))
    model_path = os.path.join(model_dir, best_model)
    print(f"使用模型: {model_path} (epoch {best_epoch})")
    
    # 加载图数据
    graph_pkl_filename = config['data'].get('graph_pkl_filename')
    _, _, adj_mx = load_graph_data(graph_pkl_filename)
    
    # 创建supervisor并加载模型
    supervisor = DCRNNSupervisor(adj_mx=adj_mx, **config)
    supervisor.load_model(model_path)
    
    # 评估
    outputs = supervisor.evaluate('test')
    
    # 计算指标
    y_true = outputs['truth']
    y_pred = outputs['prediction']
    
    # 反归一化
    scaler = supervisor._data['scaler']
    y_true = scaler.inverse_transform(y_true)
    y_pred = scaler.inverse_transform(y_pred)
    
    # 计算MAE, RMSE, MAPE, R2
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # MAPE - 过滤小值
    mask = y_true > 10
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0
    
    # R2
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    print(f"\n结果:")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²:   {r2:.4f}")
    
    return {
        'model': 'DCRNN',
        'dataset': 'PeMS08',
        'horizon': '12-step',
        'epoch': best_epoch,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    }

# ============================================================================
# 2. 评估 DSTAGNN 模型
# ============================================================================
def evaluate_dstagnn(dataset, num_for_predict, config_name):
    print("=" * 60)
    print(f"评估 DSTAGNN {dataset} {num_for_predict}-step")
    print("=" * 60)
    
    sys.path.insert(0, '/data_ssd/other_models/baseline_models/DSTAGNN')
    os.chdir('/data_ssd/other_models/baseline_models/DSTAGNN')
    
    import configparser
    from lib.utils1 import load_graphdata_channel1, compute_val_loss_mstgcn, predict_and_save_results_mstgcn
    from model.DSTAGNN import make_model
    
    # 加载配置
    config = configparser.ConfigParser()
    config.read(f'configurations/{config_name}')
    
    # 参数
    num_of_vertices = int(config['Data']['num_of_vertices'])
    points_per_hour = int(config['Data']['points_per_hour'])
    num_of_hours = int(config['Training']['num_of_hours'])
    num_of_days = int(config['Training']['num_of_days'])
    num_of_weeks = int(config['Training']['num_of_weeks'])
    batch_size = int(config['Training']['batch_size'])
    in_channels = int(config['Training']['in_channels'])
    nb_chev_filter = int(config['Architecture']['nb_chev_filter'])
    nb_time_filter = int(config['Architecture']['nb_time_filter'])
    nb_block = int(config['Architecture']['nb_block'])
    K = int(config['Architecture']['K'])
    time_strides = int(config['Architecture']['time_strides'])
    num_for_predict_cfg = int(config['Architecture']['num_for_predict'])
    
    # 使用传入的num_for_predict
    num_for_predict = num_for_predict
    
    # 设备
    ctx = config['Training']['ctx']
    os.environ["CUDA_VISIBLE_DEVICES"] = ctx
    DEVICE = torch.device('cuda:0')
    
    # 加载数据
    adj_filename = config['Data']['adj_filename']
    graph_signal_matrix_filename = config['Data']['graph_signal_matrix_filename']
    
    adj_mx = np.load(adj_filename)
    if len(adj_mx.shape) == 3:
        adj_mx = adj_mx[0]  # 取第一个邻接矩阵
    
    # 加载测试数据
    dataloader = load_graphdata_channel1(
        graph_signal_matrix_filename,
        num_of_hours, num_of_days, num_of_weeks,
        DEVICE, batch_size, shuffle=False,
        num_for_predict=num_for_predict
    )
    test_loader = dataloader['test_loader']
    scaler = dataloader['scaler']
    
    # 找到最佳模型
    if dataset == 'METR-LA':
        model_dir = f'myexperiments/METR-LA/dstagnn_h1d0w0_channel1_{num_for_predict}step_1.000000e-04'
    else:
        model_dir = f'myexperiments/{dataset}/dstagnn_h1d0w0_channel1_{num_for_predict}step_1.000000e-04'
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.params')]
    if not model_files:
        print(f"没有找到模型文件在 {model_dir}")
        return None
    
    # 选择最新的模型
    model_files.sort(key=lambda x: int(x.replace('epoch_', '').replace('.params', '')))
    best_model = model_files[-1]
    best_epoch = int(best_model.replace('epoch_', '').replace('.params', ''))
    model_path = os.path.join(model_dir, best_model)
    print(f"使用模型: {model_path} (epoch {best_epoch})")
    
    # 创建模型
    model = make_model(
        DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter,
        time_strides, adj_mx, num_for_predict, num_of_vertices,
        num_of_hours * points_per_hour
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # 评估
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            encoder_inputs, labels = batch_data
            outputs = model(encoder_inputs)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
    
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    
    # 模型输出已经是原始尺度，不需要反归一化
    # 但目标需要反归一化
    y_true = scaler.inverse_transform(y_true)
    
    # 计算指标
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # MAPE
    if dataset == 'METR-LA':
        threshold = 1.0  # 速度数据阈值
    else:
        threshold = 10.0  # 流量数据阈值
    
    mask = y_true > threshold
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0
    
    # R2
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    print(f"\n结果:")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²:   {r2:.4f}")
    
    return {
        'model': 'DSTAGNN',
        'dataset': dataset,
        'horizon': f'{num_for_predict}-step',
        'epoch': best_epoch,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    }


if __name__ == '__main__':
    results = []
    
    # 1. DCRNN PeMS08 12-step
    try:
        r = evaluate_dcrnn_pems08_12step()
        if r:
            results.append(r)
    except Exception as e:
        print(f"DCRNN PeMS08 12-step 评估失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. DSTAGNN PeMS08 12-step
    try:
        r = evaluate_dstagnn('PEMS08', 12, 'PEMS08_dstagnn_12step.conf')
        if r:
            results.append(r)
    except Exception as e:
        print(f"DSTAGNN PeMS08 12-step 评估失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. DSTAGNN METR-LA 3-step
    try:
        r = evaluate_dstagnn('METR-LA', 3, 'METR_LA_dstagnn_3step.conf')
        if r:
            results.append(r)
    except Exception as e:
        print(f"DSTAGNN METR-LA 3-step 评估失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. DSTAGNN METR-LA 6-step
    try:
        r = evaluate_dstagnn('METR-LA', 6, 'METR_LA_dstagnn_6step.conf')
        if r:
            results.append(r)
    except Exception as e:
        print(f"DSTAGNN METR-LA 6-step 评估失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. DSTAGNN METR-LA 12-step
    try:
        r = evaluate_dstagnn('METR-LA', 12, 'METR_LA_dstagnn_12step.conf')
        if r:
            results.append(r)
    except Exception as e:
        print(f"DSTAGNN METR-LA 12-step 评估失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 打印汇总
    print("\n" + "=" * 80)
    print("评估结果汇总")
    print("=" * 80)
    print(f"{'Model':<12} {'Dataset':<10} {'Horizon':<10} {'Epoch':<8} {'MAE':<10} {'RMSE':<10} {'MAPE(%)':<10} {'R²':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['model']:<12} {r['dataset']:<10} {r['horizon']:<10} {r['epoch']:<8} {r['mae']:<10.2f} {r['rmse']:<10.2f} {r['mape']:<10.2f} {r['r2']:<10.4f}")
