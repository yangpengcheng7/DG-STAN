# DG-STAN: A Dynamic Graph Spatio-Temporal Attention Network for Traffic Flow Prediction

This repository provides an implementation of **DG-STAN**, a spatio-temporal graph neural network (STGNN) for traffic forecasting. DG-STAN is designed to better model **time-varying spatial dependencies** and **multi-scale temporal patterns** in traffic data.

**Core contributions :**
- **Dynamic graph generation** from traffic states to capture time-varying spatial relations.
- **Static–dynamic fusion with node-pair-wise gating** to combine topology priors and learned dynamic relations.
- **Multi-scale temporal convolution** with kernel sizes **3 / 5 / 11** to capture short-/mid-/long-range temporal patterns within the input window.
- **Global (joint) spatio-temporal attention** to enhance long-range interactions across both nodes and time.

The paper evaluates DG-STAN on **PeMS04**, **PeMS08** (traffic flow), and **METR-LA** (traffic speed), using an input window of **12 time steps** (1 hour at 5-min intervals) and prediction horizons of **3 / 6 / 12 steps** (15 / 30 / 60 minutes).

---

## 1. Method Summary

### 1.1 Static Graph Construction
A static adjacency matrix is constructed from sensor distances using a Gaussian kernel with thresholding, then symmetrized to enforce bidirectional consistency.

### 1.2 Dynamic Graph Generation
Given an input window of node features, DG-STAN computes node representations and generates a **dynamic adjacency** via learnable projections and similarity, followed by Softmax normalization. This enables the graph structure to adapt over time.

### 1.3 Static–Dynamic Graph Fusion (Gating)
DG-STAN fuses static adjacency **A** and dynamic adjacency **A_dyn** with a **node-pair-wise gating matrix** **G**:
- Gate network (MLP): `Linear(2→16) → ReLU → Linear(16→1) → Sigmoid`
- Fused adjacency: `A_fused = G ⊙ A + (1 − G) ⊙ A_dyn`

### 1.4 Multi-Scale Temporal Convolution
DG-STAN employs three parallel temporal convolution branches with kernel sizes:
- `K = 3`, `K = 5`, `K = 11`

These branches capture short-term fluctuations, periodic/medium-term patterns, and broader temporal dependencies within the input window. Outputs are fused with learnable weighting.

### 1.5 Global Spatio-Temporal Attention
DG-STAN applies multi-head attention across:
- **Spatial** dimension (across nodes for each time step)
- **Temporal** dimension (across time for each node)

The attention streams are fused (with residual connections) to strengthen global spatio-temporal interactions.

---

## 2. Datasets and Settings

### 2.1 Datasets
Benchmarks used in the paper:
- **PeMS04**: traffic flow (5-min interval)
- **PeMS08**: traffic flow (5-min interval)
- **METR-LA**: traffic speed (5-min interval)

### 2.2 Task Setting (Paper)
- Input length: **12** time steps
- Prediction horizons: **3 / 6 / 12** steps (15 / 30 / 60 minutes)

### 2.3 Metrics
Traffic forecasting is typically reported using:
- **MAE**, **RMSE**, **MAPE**  
(Please refer to the scripts in this repo for the exact metrics computed.)

---

## 3. Repository Structure 

This repository is organized as a script-based project (most files are in the root directory):

```text
DG-STAN/
  ├── DG-STAN.py                         # DG-STAN model (main)
  ├── DG-STAN-v2.py                      # DG-STAN variant / updated version
  ├── train_dgstan_multi_horizon.py      # training script (multi-horizon)
  ├── evaluate_all_current.py            # evaluation utilities
  ├── collect_results.py                 # collect/aggregate results
  ├── ablation_attention_experiment.py   # ablation experiments
  ├── experiment_peak_offpeak.py         # peak vs off-peak analysis
  ├── prepare_stsgcn_pems04.py           # dataset preparation helper
  ├── visualize_dynamic_graph.py         # dynamic graph visualization
  ├── visualize_dynamic_graph_v2.py      # visualization (updated)
  ├── redesign_figure1.py                # figure generation
  ├── redesign_figures.py                # figure generation
  ├── TRAINING_STATUS.md                 # training progress notes
  ├── training_status_summary.md         # training summary
  ├── code_analysis_report.md            # code analysis notes
  ├── FINAL_ANALYSIS_REPORT.md           # final analysis report
  ├── staeformer_train.log               # training log (optional)
  └── README.md
```
## 4. Data Layout
If datasets are included in this repository, we recommend placing them under:

```text

data/
  ├── PeMS04/
  ├── PeMS08/
  └── METR-LA/
```
If you store datasets elsewhere, please set the dataset path through script arguments or configuration variables in the training/evaluation scripts.

## 5. Environment
Paper-reported environment (for reference):

Python 3.10

PyTorch 2.2

GPU: RTX 3090 (24GB) on Ubuntu 20.04

You may run on other platforms (Windows/Linux) with appropriate dependency adjustments.

## 6. Quick Start (Reproducibility)
The exact CLI arguments depend on the scripts in this repository.
A typical workflow is:

6.1 Install dependencies

```pip install -r requirements.txt```
Otherwise, ensure PyTorch and common scientific packages are installed.

6.2 Prepare data
Place datasets under ./data/ (recommended) or configure a custom data path in scripts.

6.3 Train
```python train_dgstan_multi_horizon.py```
6.4 Evaluate
```python evaluate_all_current.py```
6.5 Collect results
```python collect_results.py```
6.6 Visualize dynamic graph (optional)
```python visualize_dynamic_graph.py```
## 7. Citation

If you use this repository in academic work, please cite the paper:

DG-STAN: A Dynamic Graph Spatio-Temporal Attention Network for Traffic Flow Prediction

(You can add a BibTeX entry here once the paper has a final venue/DOI.)

## 8. License

Please choose a permissive license (e.g., MIT or Apache-2.0) to support reproducibility and reuse.

## 9. Notes

Line-ending warnings (LF/CRLF) may appear on Windows; they do not affect correctness.

For large datasets, consider Git LFS or GitHub Releases if pushing fails due to size limits.