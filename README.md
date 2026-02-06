# DG-STAN: A Dynamic Graph Spatio-Temporal Attention Network for Traffic Flow Prediction

This repository contains the implementation of **DG-STAN**, a spatio-temporal graph neural network (STGNN) for traffic forecasting.  
DG-STAN improves prediction accuracy by combining:

- **Dynamic graph generation** (traffic-state-based adjacency)
- **Static–dynamic graph fusion with node-pair-wise gating**
- **Multi-scale temporal convolution** (kernel sizes **3, 5, 11**)
- **Global (joint) spatio-temporal attention**

The model is evaluated on **PeMS04**, **PeMS08** (traffic flow), and **METR-LA** (traffic speed), with prediction horizons of **3/6/12 steps** (15/30/60 minutes at 5-min intervals).

---

## 1. Method Summary

### 1.1 Static Graph Construction
A static adjacency matrix is constructed from sensor distances using a Gaussian kernel with thresholding, then symmetrized to enforce bidirectional consistency.

### 1.2 Dynamic Graph Generation
For each input window, DG-STAN generates a **dynamic adjacency** based on node feature similarity after learnable projections, reflecting time-varying traffic relations.

### 1.3 Graph Gating Fusion (Static + Dynamic)
DG-STAN fuses static and dynamic graphs with a **node-pair-wise gating matrix** `G` computed by an MLP:
- MLP: `Linear(2→16) → ReLU → Linear(16→1) → Sigmoid`
- Fused adjacency: `A_fused = G ⊙ A + (1 − G) ⊙ A_dyn`

### 1.4 Multi-Scale Temporal Convolution
Three parallel temporal convolution branches with kernel sizes:
- `K1 = 3`, `K2 = 5`, `K3 = 11`

They capture short-term fluctuations, periodic/medium-term patterns, and longer dependencies within the input window.  
A learnable fusion mechanism weights different temporal scales.

### 1.5 Global Spatio-Temporal Attention
DG-STAN applies multi-head attention across:
- **Nodes (spatial attention)** per time step
- **Time (temporal attention)** per node  
Then fuses both streams with residual connections to strengthen global spatio-temporal interactions.

---

## 2. Datasets

Benchmarks used in the paper:
- **PeMS04**: traffic flow (5-min interval)
- **PeMS08**: traffic flow (5-min interval)
- **METR-LA**: traffic speed (5-min interval)

Input/Output setting (paper):
- Input window: **12 time steps** (1 hour)
- Prediction horizon: **3 / 6 / 12 steps** (15 / 30 / 60 minutes)

---

## 3. Environment

Paper-reported environment:
- Python 3.10
- PyTorch 2.2
- (Paper runs on Ubuntu + RTX 3090; Windows is also supported with path adjustments)

---

## 4. Local Paths (Your Current Setup)

- Code (this repo):
  `F:\BaiduNetdiskDownload\other_models\code`

- Data (local, do NOT recommend uploading to GitHub):
  `F:\BaiduNetdiskDownload\other_models\data`

> Tips: Keep datasets outside the repo and set dataset path in scripts/configs accordingly.

---

## 5. Running (Example)

Please check the scripts in this repo for arguments and dataset path settings. Typical workflow:

1. Prepare datasets under your local data directory
2. Train:
   - `train_dgstan_multi_horizon.py`
3. Evaluate / collect results:
   - `evaluate_all_current.py`
   - `collect_results.py`
4. Visualization:
   - `visualize_dynamic_graph.py`

---

## 6. Notes on Reproducibility (Paper Settings)

Key hyperparameters in the paper:
- Hidden dimension: 64
- Attention heads: 4
- Dropout: 0.1
- Temporal kernels: 3, 5, 11
- Optimizer: AdamW + OneCycleLR
- Batch size: 64
- Max epochs: 200 (early stopping patience 10)

---

## 7. Citation

If you use this code, please cite the paper:

**DG-STAN: A Dynamic Graph Spatio-Temporal Attention Network for Traffic Flow Prediction**

---

## 8. License

Choose a license (e.g., MIT) to make the repository reusable and reviewer-friendly.
