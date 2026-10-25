Key components (as implemented)

Embedding layer. Concatenates a feature embedding (convolution with squeeze-and-excitation channel attention), a time-of-day embedding, a day-of-week embedding, a spatial (node) embedding, and an adaptive (time × node) embedding.

Temporal Block. Splits the time axis into non-overlapping windows of size W and applies causal multi-head self-attention inside each window, with a learnable positional encoding. Only the representation of the last time step (plus a residual connection) is passed on. In all reported experiments T = W = 12, so there is one window and the block equals full causal attention; no complexity reduction is realized and long-sequence behaviour has not been evaluated. Information is not exchanged across windows.

Spatial Block. Each layer combines (i) multi-head self-attention across all nodes (global branch, with residual connection and layer normalization) and (ii) a node-context enhancement branch (LA), fused by a learned two-layer sigmoid gate and followed by a feed-forward network. The LA branch operates along the node axis: it projects features to a lower dimension, pools nodes with similarity-based weights (softmax over scalar node descriptors, N' = ⌊N/3⌋ pooled nodes), applies a small MLP with sigmoid, interpolates the resulting weights linearly back to N nodes, and re-weights the input (with an additional scalar sigmoid gate from the first feature channel). It does not use the road-network adjacency, its interpolation follows the node index order of the dataset, and it does not model geographic locality. Supported fusion modes: integrated (default), parallel, sequential, adaptive.

Star Block. Depthwise convolutions followed by an MLP with element-wise multiplication (star operation). Because the temporal block keeps only the last step, the features are replicated along time (12 steps) before this block; its benefit is therefore that of a multiplicative nonlinear transformation, not of additional temporal modelling. The temporal aggregation is weighted (learnable weights) for METR-LA and last for PEMS-BAY [CHECK].

1. Disclosures (read first)
Single runs. Every reported number comes from one training run per configuration with seed 42. No variance estimates were computed; small differences (e.g., 3.32 vs 3.34 MAE at 60 min on METR-LA) should not be read as significant.
Clipping before splitting. The speed channel is clipped to its 1st–99th percentile range computed on the whole series, before the split. Evaluation targets are therefore the clipped values. Baseline numbers taken from the literature were computed without this step (Sec. 7).
Normalization. Inputs are scaled with the median and interquartile range of the training inputs (robust scaling), not Z-score.
Window setting. in_steps = window_size = 12 in all experiments.
Module naming. "Local" in LA denotes a coarse pooling-and-re-weighting operation over nodes, not geographic locality.
2. Repository structure
LESTformer/
├── model/
│   ├── LESTformer.py       # main model (embedding, temporal block, spatial layer, Star)
│   ├── star.py             # Star block
│   └── LA.py               # node-context enhancement branch (LA)
├── lib/
│   ├── data_prepare.py     # preprocessing, scaler, data loaders
│   └── utils.py            # helpers, metrics (MAE / RMSE / MAPE)
├── configs/
│   └── LESTformer.yaml     # all configurations (METR-LA, PEMS-BAY, ablation variants)
├── train.py                # training and evaluation entry point
├── data/                   # METR-LA / PEMS-BAY (not included, see Sec. 3)
└── README.md

[CHECK] Working file names in the authors' code are STAEformer19_1_32xiaorongmetrla.py (model), data_prepare19_1.py, train19_1_32xiaorongmetrla9.py and STAEformer_star19_1_32xiaorongmetrla9.yaml. Rename them as above (or keep the working names) and update the imports and this section so that it matches the released files exactly. There is no separate test.py; evaluation runs at the end of train.py [CHECK].

3. Data and preprocessing

Data. METR-LA (207 sensors, 34,272 steps, Mar–Jun 2012) and PEMS-BAY (325 sensors, 52,116 steps, Jan–May 2017), 5-minute speed, from the DCRNN repository. Per node and time step the input has three features: [speed, time-of-day, day-of-week].

Place the files in data/<DATASET>/:

data.npz: array data of shape (T_total, N, 3);
index.npz: arrays train, val, test; each row is [x_begin, y_begin, y_end] (12 input steps, 12 output steps).

[FILL: how index.npz / data.npz were generated, or a link to the script.]

Split. Chronological 70% / 10% / 20% (train / validation / test).

Pipeline (get_dataloaders_from_index_data, scaler_type='robust'):

Clip the speed channel to [P1, P99] of the whole series. The bounds are printed in the training log. [FILL: METR-LA = … / … mph; PEMS-BAY = … / … mph]
Keep the features speed, time-of-day, day-of-week.
Fit a robust scaler (median, IQR) on the speed channel of the training inputs and apply it to the speed channel of the inputs. Targets stay in the original unit.
Predictions are inverse-transformed; the loss and all metrics are computed in the original unit.
4. Training and evaluation
bash
# METR-LA (full model)
python train.py -d METRLA -g 0 --seed 42
# PEMS-BAY   [CHECK: configuration, see Sec. 5]
python train.py -d PEMSBAY -g 0 --seed 42
Optimizer: Adam; loss: masked MAE; multi-step learning-rate decay (factor 0.1); gradient-norm clipping 3.0; batch size 16; at most 50 epochs; early stopping on the validation loss. The best-validation checkpoint is evaluated on the test set.
Parameter-group learning rates: base rate for most parameters, 0.9 × base for the LA / fusion-gate parameters, 0.8 × base for the Star parameters.
Metrics: MAE, RMSE, MAPE over all 12 steps and at horizons 3, 6, 12 (15, 30, 60 min), in the original unit. [FILL: state whether zero/missing values are masked in the metric code.]
Logs are written to ../logs/, checkpoints to ../saved_models/, and metrics to ../results/*.json.
Evaluation of a saved checkpoint: [FILL: command, or add a --test_only option].
5. Hyperparameters (from the configuration file)
Setting	METR-LA	PEMS-BAY [CHECK]
Learning rate / weight decay	8e-4 / 3e-4	1e-3 / 1e-4
LR milestones (decay 0.1)	20, 30	10, 30
Early-stopping patience	30	20
Batch size / max epochs	16 / 50	16 / 50
Embedding dims (feature / tod / dow / spatial / adaptive)	24 / 24 / 8 / 8 / 40	24 / 24 / 24 / 0 / 32
Total embedding dim D	104	104
Heads / spatial layers L / FFN dim	4 / 3 / 128	4 / 2 / 128
Dropout	0.1	0.1
Temporal block (depth / heads / window)	2 / 2 / 12	0 / 2 / 12
Star (MLP ratio / conv mode / aggregation)	3 / mixed / weighted	3 / mixed / last
LA branch	on, integrated	off
Embedding conv mode	enhanced_postfusion	enhanced_postfusion

[CHECK] The PEMS-BAY column is copied from the PEMSBAY section of the configuration file, which disables temporal attention and LA and uses no spatial embedding. If the reported PEMS-BAY results used a different configuration, replace this column and the YAML with the configuration actually used.

Ablation variants use their own YAML sections; their learning rates (0.0007 in the sections we checked) differ from the full model's (0.0008). [FILL: confirm and document the settings of every variant.]

6. Random seeds

The seed is set once at the start of training. Default 42, used for all reported results. [FILL: state whether cuDNN determinism is enforced; if not, results may differ slightly across hardware and library versions.]

7. Model selection and baselines

Hyperparameter selection. Selection used the validation set only. [FILL: describe how it was done (manual tuning, or grid search) and list the ranges.]

Hyperparameter	Searched values	Selected (METR-LA / PEMS-BAY)
Learning rate	[FILL]	8e-4 / 1e-3
Weight decay	[FILL]	3e-4 / 1e-4
Spatial layers L	[FILL]	3 / 2
Embedding dims	[FILL]	see Sec. 5
Window size W	[FILL]	12
Star aggregation	[FILL]	weighted / last
LA fusion mode	[FILL]	integrated

Baseline provenance. [FILL: one row per baseline; delete the options that do not apply.]

Baseline	Source of reported numbers (original paper / public benchmark / re-run)	Same split as ours?	Same preprocessing?	Same hardware?
HA, GWNet, DCRNN, AGCRN, STGCN, GTS, MTGNN, STNorm	[FILL]	[FILL]	[FILL]	[FILL]
GMAN, PDFormer, STID	[FILL]	[FILL]	[FILL]	[FILL]
STAEformer, STGformer	[FILL]	[FILL]	[FILL]	[FILL]

Numbers copied from publications were computed without the clipping step of Sec. 3, so comparisons with them are not strictly like-for-like.

8. Results

Single runs (seed 42). Best per column in bold only where LESTformer is actually best.

PEMS-BAY
Model	MAE@15	RMSE@15	MAPE@15	MAE@60	RMSE@60	MAPE@60
DCRNN	1.31	2.76	2.73%	1.97	4.60	4.68%
STAEformer	1.31	2.78	2.76%	1.88	4.34	4.41%
STGformer	1.30	2.76	2.71%	1.87	4.33	4.36%
LESTformer	1.27	2.69	2.53%	1.81	4.12	3.83%
METR-LA
Model	MAE@15	RMSE@15	MAPE@15	MAE@60	RMSE@60	MAPE@60
DCRNN	2.67	5.16	6.86%	3.54	7.47	10.32%
STAEformer	2.65	5.11	6.85%	3.34	7.02	9.70%
STGformer	2.78	5.49	7.48%	3.47	7.37	10.35%
LESTformer	2.76	5.43	6.81%	3.32	6.97	9.59%

Full tables for all baselines and the 30-minute horizon are in the paper.

Computational efficiency

Measured on an NVIDIA RTX 3090 GPU. [FILL: how training time per epoch and inference time per sample were measured (batch size, warm-up, GPU synchronization), and whether baselines ran on the same GPU.] Since T = W = 12, windowed attention equals full causal attention and cannot explain the speed-up; a likely contributor is that the temporal block outputs one time step, so spatial attention runs on one step instead of 12 (not isolated experimentally).

Data	Model	Training (s/epoch)	Inference (ms/sample)	Params (M)
PEMS-BAY	GMAN	617.6	107.67	0.88
PEMS-BAY	DCRNN	650.6	110.52	0.36
PEMS-BAY	STAEformer	246.2	23.80	1.37
PEMS-BAY	STGformer	394.8	32.73	3.84
PEMS-BAY	LESTformer	176.7	13.33	2.88
METR-LA	GMAN	393.7	136.56	3.44
METR-LA	DCRNN	396.1	26.19	1.42
METR-LA	STAEformer	182.3	17.33	4.24
METR-LA	STGformer	167.2	13.38	3.84
METR-LA	LESTformer	118.7	8.61	2.20
9. Environment

Python [FILL], PyTorch [FILL], CUDA [FILL], NVIDIA RTX 3090.

bash
pip install torch torchinfo numpy pyyaml
10. Reproducibility checklist (response to reviewer comment 4)
Requested item	Where
Implementation	model/, lib/, train.py (Sec. 2)
Complete hyperparameter settings	Sec. 5 and configs/
Preprocessing procedure	Sec. 3
Random seeds	Sec. 6 (seed 42, single runs)
Evaluation scripts	Sec. 4
Model-selection (grid-search) ranges	Sec. 7
Baseline provenance (reproduced vs. taken from papers)	Sec. 7
Citation

[FILL: citation entry after acceptance]

Acknowledgements

This research was funded by the National Natural Science Foundation of China (No. 61741216, 61402367) and the Shaanxi Provincial Natural Science Foundation (No. 2025JC-YBMS-722).

License

This project is released for academic research purposes only.
