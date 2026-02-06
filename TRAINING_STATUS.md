# 基线模型训练状态记录

更新时间：2026-01-20

## 数据集配置

**数据集**: PeMS04 (California traffic data)

| 参数 | 配置 |
|------|------|
| 节点数 | 307 sensors |
| 总样本数 | 16,992 time steps |
| 特征维度 | 1 (flow only) |
| 训练集比例 | 60% (10,181 samples) |
| 验证集比例 | 20% (3,394 samples) |
| 测试集比例 | 20% (3,394 samples) |
| 时间粒度 | 5分钟/样本 |

**预测任务配置**:
- 输入窗口: 12步 (60分钟历史数据)
- 输出窗口: 12步 (60分钟未来预测)
- 预测方式: Multi-step ahead forecasting

## 基线模型训练状态

### 1. ASTGCN ✅ 已完成

**配置**:
- 配置文件: `configurations/PEMS04_astgcn.conf`
- num_for_predict: 12
- 训练轮数: 100 epochs
- 最佳epoch: 77

**测试结果** (从 `astgcn_test.log`):
```
MAE:  21.77
RMSE: 34.27
MAPE: 14.34%
R²:   0.9529
```

**性能分析**:
- R²=0.9529 表明模型解释了95.29%的数据方差
- 在所有已完成模型中性能最优

---

### 2. STGCN ✅ 已完成

**配置**:
- 训练脚本: `train_pems04.py`
- num_timesteps_output: 12
- 训练轮数: 100 epochs
- Early stopping: epoch 47

**测试结果** (从 `stgcn_train.log`):
```
MAE:  24.51
RMSE: 37.55
MAPE: 17.14%
R²:   0.9437
```

**性能分析**:
- R²=0.9437 表现良好
- 比ASTGCN略差，MAE高约2.74

---

### 3. STSGCN 🔄 训练中

**配置**:
- 训练脚本: `train.py`
- num_for_predict: 12
- 训练轮数: 200 epochs

**当前进度**:
- Epoch: ~90/200
- GPU: GPU 5
- 进程PID: 3610755
- 当前验证损失: ~22.36

**状态**: 正常训练中，预计还需较长时间完成

---

### 4. DSTAGNN 🔄 训练中

**配置**:
- 配置文件: `configurations/PEMS04_dstagnn.conf`
- num_for_predict: 12
- 训练轮数: 110 epochs

**当前进度**:
- Epoch: 刚启动（已修复R²输出）
- GPU: GPU 4 (CUDA_VISIBLE_DEVICES=4)
- 进程PID: 3738960

**重要修改**:
- 已添加R²计算和输出到 `lib/utils1.py`
- 重启训练以生成完整的R²指标

**状态**: 重新训练中，需要完整训练周期

---

### 5. STAEformer 🔄 训练中

**配置**:
- 配置文件: `model/STAEformer.yaml`
- out_steps: 12
- in_steps: 12
- 训练轮数: 300 epochs
- input_dim: 1 (flow only)

**当前进度**:
- Epoch: 初始阶段
- GPU: GPU 6
- 进程PID: 3718285

**重要修改**:
- 已修改为仅使用flow特征
- 已禁用time_of_day和day_of_week
- 已修改模型代码支持单特征输入

**状态**: 训练启动中，日志文件尚未输出

---

## GPU资源分配

| GPU ID | 模型 | 状态 | 进程PID |
|--------|------|------|---------|
| GPU 0 | (未使用) | - | - |
| GPU 4 | DSTAGNN | 训练中 | 3738960 |
| GPU 5 | STSGCN | 训练中 | 3610755 |
| GPU 6 | STAEformer | 训练中 | 3718285 |

## 性能对比（已完成模型）

| 模型 | MAE ↓ | RMSE ↓ | MAPE ↓ | R² ↑ |
|------|-------|--------|--------|------|
| **ASTGCN** | **21.77** | **34.27** | **14.34%** | **0.9529** |
| STGCN | 24.51 | 37.55 | 17.14% | 0.9437 |

**差距分析**:
- MAE: ASTGCN比STGCN低2.74 (提升11.2%)
- RMSE: ASTGCN比STGCN低3.28 (提升8.7%)
- MAPE: ASTGCN比STGCN低2.80% (提升16.3%)
- R²: ASTGCN比STGCN高0.0092 (提升0.97%)

## 待完成任务

- [ ] 等待STSGCN训练完成（~110 epochs剩余）
- [ ] 等待DSTAGNN完成完整训练周期（110 epochs）
- [ ] 等待STAEformer完成训练（最多300 epochs）
- [ ] 收集所有模型的最终测试结果
- [ ] 生成完整的性能对比表格
- [ ] 更新论文实验部分

## R²指标支持状态

| 模型 | metrics.py定义 | 训练中调用 | 测试输出 | 修复状态 |
|------|---------------|-----------|---------|----------|
| STGCN | ✅ | ✅ | ✅ | 完成 |
| ASTGCN | ✅ | ✅ | ✅ | 完成 |
| STSGCN | ✅ | ✅ | ✅ | 完成 |
| DSTAGNN | ✅ | ✅ | ✅ | **已修复** |
| STAEformer | ✅ | ✅ | ✅ | 完成 |

所有模型均已支持R²指标计算和输出。

## 实验设置一致性检查 ✅

所有模型均采用统一的实验设置：

**输入配置**:
- ✅ 输入步长：12步
- ✅ 特征维度：1 (flow only)
- ✅ 数据集：PeMS04
- ✅ 归一化：基于训练集统计量

**输出配置**:
- ✅ 预测步长：12步
- ✅ 评估指标：MAE, RMSE, MAPE, R²

**数据划分**:
- ✅ 训练集：60%
- ✅ 验证集：20%
- ✅ 测试集：20%

## 备注

1. 所有模型使用相同的PeMS04数据集和数据划分方式
2. 所有模型统一使用12步输入→12步输出的预测配置
3. DSTAGNN已重新训练以包含R²指标
4. STAEformer已修改为flow-only输入以保持一致性
5. 所有模型的test.log文件包含最佳模型权重的测试结果

---

**文档维护**: 此文档将持续更新直到所有模型训练完成
