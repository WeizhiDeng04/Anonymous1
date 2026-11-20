# FROG: Experimental Settings and Reproduction

This document collects the experimental settings accompanying the paper. The released implementation uses the internal model name `DReSTG`; FROG is the name used in the manuscript.

The settings below distinguish the configuration released in this repository from experiment information reported in the manuscript. The released runner covers the main forecasting configuration. Additional protocols for individual baselines, sensitivity experiments, and seed aggregation require their corresponding experiment records.

## 1. Evaluation Metrics and Data Processing

### Metrics and training objective

Forecasting accuracy is evaluated using mean squared error (MSE) and mean absolute error (MAE). MSE is also used as the training loss.

For one dataset, forecast horizon, and random seed, the metrics average prediction errors over the evaluated forecast windows, prediction steps, and variables. The model outputs and targets passed to the evaluation routine are in the standardized data space. Overlapping forecast windows contribute separately to these averages.

The relevant implementation is provided in [exp/exp_main.py](exp/exp_main.py) and [utils/metrics.py](utils/metrics.py).

### Dataset splits and normalization

| Dataset group | Training / validation / test split |
| --- | --- |
| ETTh1 and ETTh2 | Fixed chronological boundaries: 12 / 4 / 4 blocks of 30 days at hourly frequency |
| ETTm1 and ETTm2 | The same chronological boundaries at 15-minute frequency |
| Weather, Electricity, and Traffic | Chronological split: 70% / approximately 10% / 20%; training and test sizes are rounded down and validation receives the remainder |

For ETT datasets, these fixed boundaries correspond to the conventional 3:1:1 split. Validation and test inputs include the preceding input context, while forecast targets lie within their respective splits.

Each variable is standardized using `StandardScaler`. The scaler is fitted on the training split and then applied to all splits. The released configurations enable this standardization with `scale: True`. Instance normalization within the network is a separate model operation.

See [data_provider/data_loader.py](data_provider/data_loader.py).

### Forecast windows and incomplete batches

The released data loader uses the following settings:

| Split | Shuffle | Drop incomplete final batch |
| --- | --- | --- |
| Training | Yes | Yes |
| Validation | Yes | Yes |
| Test | No | Yes |

Consequently, the final test windows are omitted whenever the test-window count is indivisible by the batch size. Validation omits a shuffle-dependent subset of windows. If the dataset contains `N` windows and the batch size is `B`, the number evaluated in a complete loader pass is `B * floor(N / B)`.

For ETTh1 and ETTh2 with input length 336, forecast horizon 720, and batch size 128, the standard split yields 2,161 test windows. The loader evaluates 2,048 and omits the final 113, approximately 5.23% of the available windows.

The exact implementation is in [data_provider/data_factory.py](data_provider/data_factory.py). Coverage for the other dataset/horizon combinations should be computed from the actual dataset files using the same window-count rule.

## 2. Common Model and Training Settings

| Setting | Value |
| --- | --- |
| Input length `seq_len` (T) | 336 |
| Forecast horizons `pred_len` | 96, 192, 336, 720 |
| Number of coarse patches `coarse_patch_count` (N_C) | 7 |
| Coarse patch length (P_C = T / N_C) | 48 |
| Fine patch overlap | Fine patch length equals twice its stride |
| Relation embedding width (D_R) | D / 4, as specified in the manuscript |
| Optimizer | Adam |
| Training objective | MSE |
| Maximum training epochs | 100 |
| Checkpoint selection | Early stopping using validation MSE; reload the selected checkpoint before test evaluation |
| `mixhop_depth` | 1 |
| `mixhop_alpha` | 0.05 |
| `gnn_r_degree` | 1 |
| `mixhop_support_preserving` | False |
| RevIN affine transformation | False |
| `theta_s_init` | 0.5413248546 |
| `boundary_eps` | 0.001 |
| `lambda_init` | 0.1 |
| Automatic mixed precision | Disabled by default |
| Data-loader workers | 0 by default |

The relation-width ratio and training hardware are manuscript-reported settings. Numerical configuration keys retain their exact code names in this document. In particular, `gnn_hidden_size` and `graph_conv_hidden_size` are separate configuration entries and should be interpreted through their use in the implementation.

Argument defaults are defined in [run.py](run.py). Dataset YAML files override those defaults, and explicit command-line arguments override the YAML values. The supplied runner explicitly fixes input length, horizon, seed, and other common options. Reproduction should therefore use the effective combination of the runner, YAML file, and argument defaults.

## 3. Dataset-Specific Configurations

### Patch settings, candidate budgets, and training

The candidate-budget parameter k_m corresponds to `softk_mass_ratio`.

| Dataset | Variables | Fine patch length | Fine stride | k_m | Batch size | Learning rate | Early-stopping patience |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | 7 | 12 | 6 | 0.50 | 128 | 0.001 | 5 |
| ETTh2 | 7 | 12 | 6 | 0.25 | 128 | 0.0001 | 10 |
| ETTm1 | 7 | 12 | 6 | 0.35 | 128 | 0.001 | 10 |
| ETTm2 | 7 | 12 | 6 | 0.65 | 128 | 0.0001 | 10 |
| Weather | 21 | 24 | 12 | 0.20 | 128 | 0.001 | 5 |
| Electricity | 321 | 12 | 6 | 0.01 | 16 | 0.0005 | 5 |
| Traffic | 862 | 16 | 8 | 0.0037 | 8 | 0.0001 | 10 |

These fine grids satisfy `P_F = 2 * S_F` and `(P_C - P_F) mod S_F = 0` with `P_C = 48`.

### Encoder settings

| Dataset | `e_layers` | `d_model` | `n_heads` | `d_ff` | `dropout` |
| --- | --- | --- | --- | --- | --- |
| ETTh1 | 2 | 32 | 8 | 128 | 0.5 |
| ETTh2 | 2 | 32 | 8 | 128 | 0.5 |
| ETTm1 | 2 | 32 | 8 | 128 | 0.5 |
| ETTm2 | 2 | 32 | 16 | 128 | 0.5 |
| Weather | 2 | 64 | 16 | 256 | 0.2 |
| Electricity | 3 | 64 | 16 | 256 | 0.2 |
| Traffic | 3 | 128 | 16 | 256 | 0.2 |

Traffic explicitly sets `lradj: TST`, `pct_start: 0.2`, `fc_dropout: 0.2`, and `head_dropout: 0.0`. The other datasets resolve to `lradj: type3` and `pct_start: 0.3` through the YAML files or argument defaults. See [utils/tools.py](utils/tools.py) for the learning-rate adjustment implementation.

The complete per-dataset configurations, including additional graph-module parameters, are provided in:

- [ETTh1](configs/ETTh1.yaml)
- [ETTh2](configs/ETTh2.yaml)
- [ETTm1](configs/ETTm1.yaml)
- [ETTm2](configs/ETTm2.yaml)
- [Weather](configs/weather.yaml)
- [Electricity](configs/electricity.yaml)
- [Traffic](configs/traffic.yaml)

## 4. Hardware and Software

The manuscript reports training on a single NVIDIA GeForce RTX 3090 GPU with 24 GB of memory. The code uses PyTorch and DGL.

The released [requirements.txt](requirements.txt) pins the following packages:

| Package | Version |
| --- | --- |
| NumPy | 1.24.3 |
| Matplotlib | 3.7.1 |
| scikit-learn | 1.3.0 |
| pandas | 2.0.1 |
| PyYAML | 6.0 |

The released dependency file leaves the Python, PyTorch, DGL, CUDA, and GPU-driver versions unspecified. The versions used for the reported experiments must be obtained from the original training environment. The listed requirements alone do not define the complete GPU training environment.

## 5. Main-Experiment Reproduction

### Data locations

Place the benchmark CSV files at the paths used by the runner:

| Dataset | Path |
| --- | --- |
| ETTh1 | `datasets/Ett_Series/ETTh1.csv` |
| ETTh2 | `datasets/Ett_Series/ETTh2.csv` |
| ETTm1 | `datasets/Ett_Series/ETTm1.csv` |
| ETTm2 | `datasets/Ett_Series/ETTm2.csv` |
| Weather | `datasets/Customs/weather.csv` |
| Electricity | `datasets/Customs/electricity.csv` |
| Traffic | `datasets/Customs/traffic.csv` |

The custom datasets use the data-loader identifier `custom` together with the explicit dataset identifier `Weather`, `Electricity`, or `Traffic`.

### Reference caches

The model requires validated reference caches for training, validation, and test splits. Cache construction, verification, and sample alignment are implemented in [scripts/ref.py](scripts/ref.py), [data_provider/ref_dataset.py](data_provider/ref_dataset.py), and [utils/ref_cache.py](utils/ref_cache.py).

The default reference root is `reference`. The runner defaults to `DRESTG_REFERENCE_MODE=reuse`, which verifies existing caches. For a fresh setup, build the caches before running the main experiments.

Run the following commands from the repository root after installing a compatible training environment and placing the datasets:

```bash
# Check the configuration and construct the 28-cell experiment plan.
bash scripts/DReSTG_run.sh 0 dry-run

# Build and validate reference caches for all seven datasets and three splits.
DRESTG_REFERENCE_MODE=build bash scripts/DReSTG_run.sh 0 prepare

# Train and evaluate the 28 main experiment cells using seed 2021.
DRESTG_REFERENCE_MODE=reuse bash scripts/DReSTG_run.sh 0 full
```

The runner supports the following modes:

| Mode | Purpose |
| --- | --- |
| `dry-run` | Configuration and experiment-plan checks |
| `prepare` | Build or verify reference caches according to the reference mode |
| `smoke` | One-epoch Traffic run at horizon 96 |
| `chain2` | One-epoch ETTh1 runs at horizons 96 and 192 |
| `full` | Main experiments for all seven datasets and four horizons |

The full runner launches ETT experiments in pairs. `ETT_SECOND_GPU` defaults to the first GPU, so two paired processes can share that GPU. If a second compatible GPU is available, its identifier can be supplied through `ETT_SECOND_GPU`. Peak memory depends on the selected configuration and concurrent processes.

Compact metrics, logs, completion records, and runtime records are collected under `drestg_results/completed_cells/`. The final aggregate file is `drestg_results/summary.tsv`. The runner invokes its cleanup script after each successful cell and removes the corresponding trained checkpoint; it retains the compact result records. See [scripts/DReSTG_cleanup.sh](scripts/DReSTG_cleanup.sh) before adapting the workflow for checkpoint retention.
