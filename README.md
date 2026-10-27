# LESTformer: Locally Enhanced Spatio-Temporal Transformer for Traffic Speed Prediction

> **Ting Tong***

---

## Overview

LESTformer is a spatio-temporal Transformer for traffic speed prediction. It combines three components: windowed temporal attention, a spatial block that couples intra-node feature-wise local enhancement with global inter-node attention, and a Star block for high-dimensional nonlinear feature transformation. The design targets the following limitations of existing traffic forecasting models:

- **Limited sensitivity to fine-grained local fluctuations** in the node feature representation
- **Insufficient fusion** of local feature patterns and global inter-node spatio-temporal dependencies

On **PEMS-BAY**, LESTformer obtains the best result among the compared baselines at every forecasting horizon, with a MAPE of **3.83%** at 60 minutes. On **METR-LA**, it obtains a MAPE of **9.59%** at 60 minutes. Its advantage on METR-LA is concentrated at longer horizons: it is **not** the best model for every metric and every horizon (for example, at 15 minutes STAEformer is better on MAE and RMSE).

> **Scope note.** All experiments use two California highway speed datasets with a 12-step input and 12-step output configuration. Generalization to other cities, other traffic variables, longer historical contexts, missing-data conditions, and distribution shifts has **not** been validated (see [Limitations](#limitations)).

---

## Architecture

<img width="1116" height="460" alt="architecture" src="https://github.com/user-attachments/assets/4212aaea-34a4-4c44-8f7f-416318be47f3" />

*Fig. 1 — Overall architecture of LESTformer. The model takes historical traffic observations X as input and produces future predictions Ŷ through four sequential components: Embedding layer, Temporal Block, Spatial Block, and Star Block.*

### Key Components

**Embedding layer** — Integrates four complementary sources into a unified representation: feature embedding (Conv2d + SE channel attention), time-of-day embedding, day-of-week embedding, and adaptive (node-time) embedding.

**Temporal Block** — Partitions the time axis into non-overlapping windows of size `W` and applies causal multi-head self-attention within each window. A learnable positional encoding is added before attention.
- The theoretical cost of windowed attention is O(T·W·D), versus O(T²·D) for full temporal attention. This is a theoretical statement only: **it has not been validated empirically on long inputs.**
- Windows are non-overlapping and the current design has **no mechanism for exchanging information across windows**; temporal dependencies are modeled only within each window.

**Spatial Block** — Each layer combines two components, fused by a learned sigmoid gate:
- *Intra-node feature-wise local enhancement* (implemented in `LA.py`, originally named "LocalAttention"). It applies SoftPool along the **feature dimension of each node**. It does **not** operate across neighboring graph nodes and therefore does **not** model inter-node spatial locality. In the concatenated feature vector (speed, time-of-day, day-of-week, and adaptive embeddings), adjacent coordinates are neighboring latent feature subspaces, not spatial neighbors. The module acts as a local feature-mixing operator that adds a local inductive bias before the linear projection.
- *Inter-node multi-head self-attention*, which alone models global spatial dependencies among nodes.


**Star Block** — Applies multi-scale depthwise convolutions along both temporal and spatial axes, followed by an MLP. Element-wise multiplication implicitly maps features into an exponentially high-dimensional nonlinear space. Learnable temporal aggregation weights select the final representation across time steps. Its advantage over standard FFN and CNN alternatives is modest.

---

## Requirements

```
Python >= 3.8
PyTorch >= 1.12
torchinfo
numpy
```

Install dependencies:

```bash
pip install torch torchinfo numpy
```

---

## Repository Structure

```
LESTformer/
├── model/
│   ├── LESTformer.py   # Main model definition
│   ├── star.py                        # Star
│   └── LA.py                          # Intra-node feature-wise local enhancement (SoftPool)
├── data/
│   └── ...                            # Dataset files (METR-LA, PEMS-BAY)
├── configs/
│   └── ...                            # Training configuration files (YAML)
├── train.py                           # Training entry point
├── test.py                            # Evaluation entry point
├── architecture.png                   # Model architecture figure
└── README.md
```

---

## Data Preparation

Download the METR-LA and PEMS-BAY datasets from the [DCRNN repository](https://github.com/liyaguang/DCRNN) and place them under the `data/` directory.

| Dataset | Nodes | Interval | Time Steps | Time Range |
|---|---|---|---|---|
| METR-LA | 207 | 5 min | 34,272 | Mar–Jun 2012 |
| PEMS-BAY | 325 | 5 min | 52,116 | Jan–May 2017 |

Input features per time step per node: `[traffic speed, time-of-day, day-of-week]`
Data split: **70% train / 10% validation / 20% test** (chronological)
Normalization: Z-score
Input / output configuration: 12 steps in, 12 steps out (T = 12)

---

## Reproducibility

The following are provided in this repository for assessing the reported results:

- **Implementation**: model definition under `model/`.
- **Hyperparameters**: complete settings in the YAML files under `configs/`.
- **Preprocessing and evaluation**: data loading, Z-score normalization, chronological 7:1:2 split, and the evaluation script `test.py`.
- **Random seeds**: the default seed is **42**. The repeated runs reported below use seeds **42, 2024, and 3407**.

### Training settings (Section 5.2 of the paper)

Adam optimizer, learning rate 0.001, batch size 16, up to 200 epochs, early stopping if the validation loss does not improve for 70 consecutive epochs. Experiments were run on a single NVIDIA RTX 3090 GPU. Final hyperparameters were selected by grid search on validation performance.

### Grid-search ranges used for model selection

## Hyperparameter Search Space

To select the optimal hyperparameters, we conducted a systematic manual grid search on the validation set. The search ranges and the final selected values for both datasets are summarized below.

| Hyperparameter | Search range | Selected (METR-LA / PEMS-BAY) |
|---|---|---|
| Learning rate | `{1e-4, 5e-4, 8e-4, 1e-3, 2e-3}` | 0.0008 / 0.001 |
| Batch size | `{8, 16, 32, 64}` | 16 / 16 |
| Input embedding dimension | `{16, 24, 32}` | 24 / 24 |
| ToD / DoW embedding dims | `{8, 16, 24}` | 24 / 8 (LA), 24 / 24 (BAY) |
| Spatial / Adaptive dims | `{0, 8, 16, 32, 40}` | 8 / 40 (LA), 8 / 32 (BAY) |
| Window size (W) | `{6, 12}` | 12 / 12 |
| Number of layers (L) | `{1, 2, 3, 4}` | 3 / 2 |
| Number of heads | `{2, 4, 8}` | 4 / 4 |
| Temporal depth | `{0, 2}` | 2 / 0 |
| Feed-forward dim | `{128, 256}` | 128 / 128 |
| Dropout rate | `{0.1, 0.15, 0.2, 0.3}` | 0.1 / 0.1 |

### Source of baseline results

Baseline results (e.g., DCRNN, STAEformer, STGformer) are **taken from their original publications** and were **not re-run** in our computational environment. They follow the same public datasets, the same chronological 7:1:2 split, and the same 12-step input / 12-step output protocol as this work. Only LESTformer (and its ablation / comparison variants) was run in our environment.

---

## Quick Start

### Training

```bash
python train.py --dataset PEMS-BAY --device cuda:0
```

### Evaluation

```bash
python test.py --dataset PEMS-BAY --checkpoint path/to/checkpoint.pth
```

---

## Results

Single-run results with seed 42. Baseline numbers are from the original publications.

### PEMS-BAY

| Model | MAE@15min | RMSE@15min | MAPE@15min | MAE@60min | RMSE@60min | MAPE@60min |
|---|---|---|---|---|---|---|
| DCRNN | 1.31 | 2.76 | 2.73% | 1.97 | 4.60 | 4.68% |
| STAEformer | 1.31 | 2.78 | 2.76% | 1.88 | 4.34 | 4.41% |
| STGformer | 1.30 | 2.76 | 2.71% | 1.87 | 4.33 | 4.36% |
| **LESTformer** | **1.27** | **2.69** | **2.53%** | **1.81** | **4.12** | **3.83%** |

### METR-LA

| Model | MAE@15min | RMSE@15min | MAPE@15min | MAE@60min | RMSE@60min | MAPE@60min |
|---|---|---|---|---|---|---|
| DCRNN | 2.67 | 5.16 | 6.86% | 3.54 | 7.47 | 10.32% |
| STAEformer | **2.65** | **5.11** | 6.85% | 3.34 | 7.02 | 9.70% |
| STGformer | 2.78 | 5.49 | 7.48% | 3.47 | 7.37 | 10.35% |
| **LESTformer** | 2.76 | 5.43 | **6.81%** | **3.32** | **6.97** | **9.59%** |

On METR-LA, LESTformer is not the best model on every metric and horizon: STAEformer is better on MAE and RMSE at 15 minutes. Claims of superiority are limited to the differences supported by the repeated runs below.

### Robustness: repeated runs

LESTformer was run three times with seeds 42, 2024, and 3407. Values are mean ± standard deviation of MAE at the 60-minute horizon.

| Dataset | (mean ± std, 3 seeds) |
| Dataset | Model | H3 MAE (15min) | H6 MAE (30min) | H12 MAE (60min) |
|---|---|---|---|---|
| PEMS-BAY | STAEformer | 1.31 ± 0.01 | 1.62 ± 0.01 | 1.88 ± 0.01 |
| PEMS-BAY | LESTformer | 1.27 ± 0.01 | 1.56 ± 0.01 | 1.81 ± 0.01 |
| METR-LA | STAEformer | 2.65 ± 0.01 | 2.97 ± 0.01 | 3.34 ± 0.01 |
| METR-LA | LESTformer | 2.76 ± 0.01 | 2.94 ± 0.01 | 3.33 ± 0.01 |
### Comparison with feature-level alternatives (Section 5.8)

The SoftPool-based intra-node feature-wise local enhancement was compared with an MLP, squeeze-and-excitation (SE), and channel attention (CA) in the same position. Result on METR-LA at 60 minutes:

| Variant | MAE@60min | RMSE@60min |
|---|---|---|
| **SoftPool (ours)** | **3.32** | **6.97** |
| MLP | 3.39 | 7.28 |
| SE | 3.39 | 7.27 |
| CA | 3.41 | 7.30 |

The differences are small (0.01–0.02 in MAE). Results for the other metrics, horizons, and PEMS-BAY are in Section 5.8 of the paper. [TODO: state whether these variants were run with a single seed or repeated.]

### Feature-axis vs. graph-adjacency neighborhood (Section 5.7)

Redefining the SoftPool neighborhood along the spatial node axis using the graph adjacency matrix (LA-on-graph) gave worse results than the feature-axis design in the paper's experiments (MAE 1.51 vs. 1.59 on PEMS-BAY; 3.12 vs. 2.94 on METR-LA, as reported in Section 5.7).

---

## Computational Efficiency

Measured on an NVIDIA RTX 3090 GPU with T = W = 12 only (Section 5.9, Table 5).

| Dataset | Model | Training (s/epoch) | Inference (ms/sample) | Params (M) |
|---|---|---|---|---|
| PEMS-BAY | STAEformer | 246.2 | 23.80 | 1.37 |
| PEMS-BAY | STGformer | 394.8 | 32.73 | 3.84 |
| PEMS-BAY | **LESTformer** | **176.7** | **13.33** | 2.88 |
| METR-LA | STAEformer | 182.3 | 17.33 | 4.24 |
| METR-LA | STGformer | 167.2 | 13.38 | 3.84 |
| METR-LA | **LESTformer** | **118.7** | **8.61** | 2.20 |

(GMAN and DCRNN are also in Table 5 of the paper.) Because T = W = 12, windowed attention is equivalent to full causal attention here, so these speed differences cannot be attributed to the windowing mechanism; LESTformer does not have the fewest parameters among the compared methods.

No scalability analysis over longer inputs (e.g., T = 24, 48, 96) or comparison with full temporal attention is reported, so no claim is made about efficiency on long sequences.

---

## Limitations

- Evaluation covers only two California highway speed datasets (METR-LA, PEMS-BAY) with a 12-step input and 12-step output configuration.
- Generalization to other cities, other traffic variables, longer historical contexts, missing-data conditions, and distribution shifts has not been validated.
- The windowed temporal attention does not exchange information across non-overlapping windows, and its efficiency benefit for long sequences has not been verified (T = W = 12 in all experiments).
- The intra-node feature-wise local enhancement models feature-axis structure within each node, not inter-node spatial locality.
- On METR-LA, LESTformer does not obtain the best result on every metric and horizon, and the advantage of the Star block over FFN and CNN alternatives is modest.

---

## Citation



---

## Acknowledgements



---

## License

This project is released for academic research purposes only.
