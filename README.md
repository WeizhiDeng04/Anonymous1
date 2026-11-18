# DReSTG/FROG

An experimental implementation of **Dual Resolution Enhanced Spatial-Temporal Graph for Long-Term Multivariate Time Series Forecasting**.

The forecasting pipeline uses continuously overlapping Fine Patch Tokens, pooled Coarse Patch Nodes, mean-pooled Global Nodes, a learned Global Graph, a statistics-guided Spatiotemporal Graph, Union candidate selection, one-hop graph propagation, residual temporal attention, and a Fine-only linear forecast decoder. The implementation follows the source-provenance notices and licenses retained in the repository.

## Repository layout

- `run.py`: frozen forecasting entry point and CLI validation
- `models/DReSTG.py`, `layers/`: forecasting architecture
- `utils/token_utils.py`, `utils/patch_utils.py`: coarse/fine token operations
- `scripts/ref.py`, `utils/ref_cache.py`: project-local Reference producer and exact verifier
- `scripts/DReSTG_run.sh`: 28-cell experiment runner
- `scripts/DReSTG_cleanup.sh`: scoped per-cell cleanup
- `configs/*.yaml`: seven frozen dataset recipes
- `statementDm.md`: detailed change, reference, runner, cleanup and verification contract

## Data

Place the actual public benchmark CSV files at:

```text
datasets/Ett_Series/{ETTh1,ETTh2,ETTm1,ETTm2}.csv
datasets/Customs/{weather,electricity,traffic}.csv
```

The committed CSVs are **placeholder stubs** and cannot be used for training or Reference generation. Keep the expected CSV headers/format of the original benchmark data.

## Reference mechanism

The default project-local Reference root is `reference/`. The cache contains:

```text
reference/base/<dataset>/<split>/{rho.npy,stat.npy,idx.npy,meta.json,ok.json}
reference/st/<dataset>/<split>/{off.npy,pair.npy,r.npy,meta.json,ok.json}
```

`base` stores full signed raw Pearson and normalized coarse-window statistics; `st` stores canonical, positive-only, cumulative-mass Soft-K packed variable pairs. `build` generates and exact-verifies each split before atomic publication. `reuse` only verifies and consumes existing assets from the same root, without rebuilding or any external-project fallback.

For a fresh clone, after supplying the real data, execute:

```bash
DRESTG_REFERENCE_MODE=build bash scripts/DReSTG_run.sh 0 prepare
```

Thereafter use the same Reference root, normally with `DRESTG_REFERENCE_MODE=reuse`. The new `drestg_*` metadata schemas require rebuilding assets produced under earlier experimental schemas; no numerical Reference method was changed. To place assets elsewhere, set `DRESTG_REFERENCE_ROOT` explicitly; use `REF_DEVICE` and `REF_BATCH` to control Reference construction resource usage (the defaults are `cpu` and `64`).

## Training and run lifecycle

```bash
bash scripts/DReSTG_run.sh 0 dry-run
bash scripts/DReSTG_run.sh 0 smoke
bash scripts/DReSTG_run.sh 0 chain2
bash scripts/DReSTG_run.sh 0 full
```

`dry-run` validates the 7×4 seed-2021 plan and model recipe without accessing data/GPU. `smoke` tests one Traffic-96 cell with 1 epoch; `chain2` tests concurrent ETTh1-96 and ETTh1-192 cells with 1 epoch; `full` runs all 28 formal cells for up to 100 epochs per cell, subject to each frozen configuration's early stopping.

`full` runs ETT datasets in groups of **two concurrent cells**, completing and validating both before advancing. Weather, Electricity and Traffic run **strictly one cell at a time**, after all ETT cells have finished. The first GPU ID is the first shell argument. `ETT_SECOND_GPU` optionally assigns a separate GPU to the second ETT process; when omitted it shares the first GPU, preserving the source runner's GPU assignment behavior. Ensure sufficient GPU memory before using same-device parallelism. This scheduling option does not alter training hyperparameters.

A single external `screen` session may be started manually:

```bash
screen -S drestg_full
bash scripts/DReSTG_run.sh 0 full 2>&1 | tee drestg_full.log
```

The runner never creates/manages screen sessions. Every cell retains compact `metrics.json`, `runtime.json`, `formal.log` and `complete.json`. Checkpoints and redundant result/test arrays are cleaned **only after successful result validation**; failed cells retain forensic information and stop the matrix. Completed cells are idempotently skipped after identity/recipe verification. The final `drestg_results/summary.tsv` requires all 28 verified cells.

## Frozen modeling and experiment contract

`T=336`, `N_C=7`, `P_C=48`, `S_F=P_F/2`, horizons `96/192/336/720`, seed `2021`, original seven per-dataset YAML hyperparameters, `mixhop_depth=1`, `mixhop_alpha=0.05`, initialization scalars `theta_s_init=0.5413248546`, `boundary_eps=0.001`, `lambda_init=0.1`. The variable count, embedding dimension, patch size, batch size, learning rate, scheduler, dropout and Soft-K mass budget are read from their frozen YAML files. No automatic missing-Reference fallback exists.

For detailed source mapping, compatibility limits and audit test levels, consult `statementDm.md`.

## Licensing and provenance

Honor the root `LICENSE` and copyright notices within modified and borrowed files, including the original PatchTST-related notices. Reusing this repository requires compliance with any applicable upstream licenses.
