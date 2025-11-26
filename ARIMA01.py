# # -*- coding: utf-8 -*-
# """
# 基于 ARIMA 的 PEMS04 时间序列预测
# 环境要求：Python 3.8+, pandas, numpy, pmdarima, scikit-learn
# """
# import numpy as np
# import pandas as pd
# from pmdarima import auto_arima
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# import warnings
# warnings.filterwarnings('ignore')  # 禁用警告信息
#
# # ---------------------- 数据处理 ----------------------
# def load_pems_data(npz_path):
#     """加载数据集并转换为合适维度"""
#     data = np.load(npz_path)
#     flow_data = data['data']  # [总时间步, 节点数, 特征数]
#     return flow_data.transpose(1, 0, 2)  # 转换为 [节点数, 总时间步, 特征数]
#
# # 加载数据
# pems_data = load_pems_data("E:\交通流预测研究\other_models\数据集\PeMS04\PeMS04.npz")
# num_nodes, total_timesteps, num_features = pems_data.shape
# print(f"数据维度：节点数={num_nodes}, 时间步={total_timesteps}, 特征数={num_features}")
#
# # ---------------------- 划分数据集 ----------------------
# def split_dataset(data, split_ratio=0.8):
#     """按时间轴划分训练集和测试集"""
#     split_idx = int(data.shape[1] * split_ratio)
#     return data[:, :split_idx, :], data[:, split_idx:, :]
#
# train_data, test_data = split_dataset(pems_data)
# print(f"训练集维度: {train_data.shape}, 测试集维度: {test_data.shape}")
#
# # ---------------------- ARIMA 预测 ----------------------
# def arima_forecast(train_series, test_series):
#     """自动ARIMA模型训练与预测"""
#     model = auto_arima(
#         train_series,
#         seasonal=False,
#         trace=False,
#         error_action='ignore',
#         suppress_warnings=True
#     )
#     forecast = model.predict(n_periods=len(test_series))
#     return forecast
#
# # 容器存储预测结果
# test_timesteps = test_data.shape[1]
# predictions = np.zeros((num_nodes, test_timesteps, num_features))
#
# # 对每个节点和特征进行独立预测
# for node in range(num_nodes):
#     for feature in range(num_features):
#         train_series = train_data[node, :, feature]  # 提取训练序列
#         test_series = test_data[node, :, feature]    # 提取测试序列
#
#         # ARIMA预测
#         pred = arima_forecast(train_series, test_series)
#         predictions[node, :, feature] = pred
#
#     print(f"节点 {node+1}/{num_nodes} 预测完成")
#
# # ---------------------- 评估指标 ----------------------
# def calculate_metrics(true, pred):
#     """计算评估指标"""
#     epsilon = 1e-6
#     mae = mean_absolute_error(true, pred)
#     rmse = np.sqrt(mean_squared_error(true, pred))
#     mape = np.mean(np.abs((true - pred) / (true + epsilon))) * 100
#     ss_res = np.sum((true - pred)**2)
#     ss_tot = np.sum((true - np.mean(true))**2)
#     r2 = 1 - (ss_res / (ss_tot + epsilon))
#     return mae, rmse, mape, r2
#
# # 展平数据计算全局指标
# true_flat = test_data.reshape(-1)
# pred_flat = predictions.reshape(-1)
# valid_mask = ~np.isnan(true_flat)  # 处理可能的NaN值
#
# mae, rmse, mape, r2 = calculate_metrics(
#     true_flat[valid_mask],
#     pred_flat[valid_mask]
# )
#
# # ---------------------- 打印结果 ----------------------
# print("\n评估结果（所有节点和特征的平均值）：")
# print(f"MAE: {mae:.4f}")
# print(f"RMSE: {rmse:.4f}")
# print(f"MAPE: {mape:.2f}%")
# print(f"R² Score: {r2:.4f}")
# # -*- coding: utf-8 -*-
# """
# 基于 ARIMA 的 PEMS04 时间序列预测
# 环境要求：Python 3.8+, pandas, numpy, pmdarima, scikit-learn, matplotlib, logging
# """
# # -*- coding: utf-8 -*-
# """
# 基于 ARIMA 的 PEMS04 时间序列预测（分批次处理版本）
# 环境要求：Python 3.8+, pandas, numpy, pmdarima, scikit-learn
# """

# -*- coding: utf-8 -*-
"""
基于 ARIMA 的 PEMS04 时间序列预测（超轻量版）
解决内存溢出问题：更小批次+更简化模型+实时内存释放
"""
import numpy as np
import gc
import time
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings('ignore')


# ---------------------- 数据处理 ----------------------
def load_pems_data(npz_path):
    data = np.load(npz_path)
    flow_data = data['data']  # [总时间步, 节点数, 特征数]
    return flow_data.transpose(1, 0, 2)  # [节点数, 总时间步, 特征数]


# 加载数据（替换为实际路径）
pems_data = load_pems_data("E:/交通流预测研究/other_models/数据集/PeMS04/PeMS04.npz")
num_nodes, total_timesteps, num_features = pems_data.shape
print(f"数据维度：节点数={num_nodes}, 时间步={total_timesteps}, 特征数={num_features}")


# ---------------------- 划分数据集（可选：缩短序列长度） ----------------------
def split_dataset(data, split_ratio=0.8, max_timesteps=None):
    """增加可选参数：限制最大时间步，减少序列长度"""
    split_idx = int(data.shape[1] * split_ratio)
    train = data[:, :split_idx, :]
    test = data[:, split_idx:, :]

    # 如果设置了最大时间步，截断序列（降低模型复杂度）
    if max_timesteps:
        train = train[:, -max_timesteps:, :]  # 只取最近的max_timesteps个时间步
    return train, test


# 关键优化：限制训练序列长度（例如只保留3000个时间步，原13593）
train_data, test_data = split_dataset(pems_data, split_ratio=0.8, max_timesteps=3000)
print(f"训练集维度: {train_data.shape}, 测试集维度: {test_data.shape}")


# ---------------------- 简化版ARIMA预测 ----------------------
def arima_forecast(train_series, test_series):
    """极度简化的ARIMA模型，最小化计算量"""
    model = auto_arima(
        train_series,
        seasonal=False,
        trace=False,
        error_action='ignore',
        suppress_warnings=True,
        max_p=2,  # 进一步降低AR阶数
        max_q=2,  # 进一步降低MA阶数
        max_d=1,  # 差分阶数固定为1
        stepwise=True,
        n_jobs=1,  # 单线程运行（避免多线程内存冲突）
        maxiter=50  # 限制迭代次数
    )
    forecast = model.predict(n_periods=len(test_series))
    return forecast


# ---------------------- 分批次处理（极小批次） ----------------------
test_timesteps = test_data.shape[1]
predictions = np.zeros((num_nodes, test_timesteps, num_features))

# 核心优化：批次大小降至5（根据内存情况可再减至1-3）
batch_size = 1
num_batches = (num_nodes + batch_size - 1) // batch_size

for batch in range(num_batches):
    start_node = batch * batch_size
    end_node = min((batch + 1) * batch_size, num_nodes)
    print(f"\n===== 处理批次 {batch + 1}/{num_batches}（节点 {start_node + 1}-{end_node}） =====")

    for node in range(start_node, end_node):
        # 每个节点处理前先释放一次内存
        gc.collect()
        time.sleep(1)  # 暂停1秒，给系统回收资源的时间

        for feature in range(num_features):
            # 提取序列（转换为一维数组，减少内存占用）
            train_series = train_data[node, :, feature].flatten()
            test_series = test_data[node, :, feature].flatten()

            # 预测并保存结果
            pred = arima_forecast(train_series, test_series)
            predictions[node, :, feature] = pred

        print(f"节点 {node + 1}/{num_nodes} 预测完成")

    # 批次结束强制释放内存+延迟
    gc.collect()
    time.sleep(2)  # 批次间暂停2秒，缓解内存压力


# ---------------------- 评估指标 ----------------------
def calculate_metrics(true, pred):
    epsilon = 1e-6
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    mape = np.mean(np.abs((true - pred) / (true + epsilon))) * 100
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + epsilon))
    return mae, rmse, mape, r2


# 计算并打印结果
true_flat = test_data.reshape(-1)
pred_flat = predictions.reshape(-1)
valid_mask = ~np.isnan(true_flat)

mae, rmse, mape, r2 = calculate_metrics(
    true_flat[valid_mask],
    pred_flat[valid_mask]
)

print("\n===== 评估结果 =====")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"R² Score: {r2:.4f}")
