# LESTformer: Locally Enhanced Spatio-Temporal Transformer for Traffic Speed Prediction

> **Ting Tong***


---

## Overview

LESTformer is a novel spatio-temporal Transformer architecture for traffic speed prediction. Built upon STAEformer, it introduces three tightly integrated innovations — windowed temporal attention, locally-aware spatial attention, and high-dimensional nonlinear feature transformation — to jointly address the key limitations of existing traffic forecasting models:

- **Limited sensitivity to local fluctuations** in traffic patterns
- **Quadratic computational complexity** of standard Transformers over long sequences
- **Inadequate deep fusion** of local and global spatio-temporal features

On **PEMS-BAY**, LESTformer outperforms all baselines across every forecasting horizon, achieving a MAPE of **3.83%** at 60 minutes. On **METR-LA**, it achieves a MAPE of **9.59%** at the 60-minute horizon, with especially strong performance in long-term forecasting.

---

## Architecture

<img width="1116" height="460" alt="architecture" src="https://github.com/user-attachments/assets/4212aaea-34a4-4c44-8f7f-416318be47f3" />

*Fig. 1 — Overall architecture of LESTformer. The model takes historical traffic observations X as input and produces future predictions Ŷ through four sequential components: Embedding layer, Temporal Block, Spatial Block, and Star Block.*

### Key Components

**Embedding layer** — Integrates four complementary sources into a unified representation: feature embedding (Conv2d + SE channel attention), time-of-day embedding, day-of-week embedding, and adaptive (node-time) embedding.

**Temporal Block** — Partitions the time axis into non-overlapping windows of size `W` and applies causal multi-head self-attention within each window. Reduces temporal complexity from O(T²·D) to O(T·W·D). A learnable positional encoding is added before attention.

**Spatial Block** — Embeds local importance modeling directly inside the spatial attention layer. A LocalAttention module (SoftPool along the feature dimension) captures fine-grained per-node patterns; a standard multi-head attention models global inter-node correlations; both are fused via a learned sigmoid gating mechanism. Supports four fusion modes: `integrated` (default), `parallel`, `sequential`, and `adaptive`.

**Star Block** — Applies multi-scale depthwise convolutions along both temporal and spatial axes, followed by an MLP. Element-wise multiplication implicitly maps features into an exponentially high-dimensional nonlinear space. Learnable temporal aggregation weights select the final representation across time steps.

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
│   └── LA.py                          # LocalAttention
├── data/
│   └── ...                            # Dataset files (METR-LA, PEMS-BAY)
├── configs/
│   └── ...                            # Training configuration files
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

---

## Quick Start

### Verify the model

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
| STAEformer | 2.65 | 5.11 | 6.85% | 3.34 | 7.02 | 9.70% |
| STGformer | 2.78 | 5.49 | 7.48% | 3.47 | 7.37 | 10.35% |
| **LESTformer** | **2.76** | **5.43** | **6.81%** | **3.32** | **6.97** | **9.59%** |

---

## Computational Efficiency

Measured on an NVIDIA RTX 3090 GPU.


---

## Citation



---

## Acknowledgements



---

## License

This project is released for academic research purposes only.
