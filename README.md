# DG-STAN: A Dynamic Graph Spatio-Temporal Attention Network for Traffic Flow Prediction

This repository provides the implementation of **DG-STAN**, a spatio-temporal graph neural network (STGNN) for traffic forecasting.  
DG-STAN combines:
- Dynamic graph generation (traffic-state-based adjacency)
- Static–dynamic graph fusion with node-pair-wise gating
- Multi-scale temporal convolution (kernel sizes **3, 5, 11**)
- Global (joint) spatio-temporal attention

The paper evaluates DG-STAN on **PeMS04**, **PeMS08** (traffic flow) and **METR-LA** (traffic speed), with prediction horizons **3/6/12 steps** (15/30/60 minutes at 5-min intervals).

---

## 1. Directory Layout

Recommended structure:

DG-STAN/
├── data/ # datasets (optional if you upload data)
│ ├── PeMS04/
│ ├── PeMS08/
│ └── METR-LA/
├── checkpoints/ # saved models (optional)
├── logs/ # training logs (optional)
├── *.py # source code
└── README.md

### Using a custom data path
If you store datasets elsewhere, please set the dataset path through script arguments or configuration variables.

---

## 2. Environment

- Python 3.10+
- PyTorch 2.x

(Adjust as needed for your platform.)

---

## 3. Running (Example)

Please check each script for arguments and dataset path settings. A typical workflow:

1. Prepare datasets under `./data/`
2. Train:
   - `train_dgstan_multi_horizon.py`
3. Evaluate / collect results:
   - `evaluate_all_current.py`
   - `collect_results.py`
4. Visualization:
   - `visualize_dynamic_graph.py`

---

## 4. Paper-Consistent Settings (Key)

Common settings in the paper include:
- Hidden dimension: 64
- Attention heads: 4
- Dropout: 0.1
- Temporal kernels: 3, 5, 11
- Optimizer: AdamW + OneCycleLR
- Batch size: 64
- Max epochs: 200 (early stopping patience 10)

---

## 5. Citation

If you use this code, please cite the paper:

**DG-STAN: A Dynamic Graph Spatio-Temporal Attention Network for Traffic Flow Prediction**

---

## 6. License

Choose a permissive license (e.g., MIT) for reproducibility and reuse.
