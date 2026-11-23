# DReSTG (its new name is FROG)

## Code Structure

- `run.py`: model execution entry point and configuration loading.
- `models/DReSTG.py`: forecasting model interface.
- `layers/DReSTG_backbone.py`: fine patch embedding, coarse node construction, temporal encoding, and forecasting head.
- `layers/TemporalEncoder.py`: temporal attention and spatiotemporal graph encoding.
- `layers/DualResolutionGraph.py`: global relation estimation, candidate union, and coarse-node enhancement.
- `layers/SpatiotemporalRelation.py`: statistical relation weighting.
- `layers/GraphPropagation.py`: one-hop graph propagation.
- `layers/ForecastDecoder.py`, `layers/PatchTST_layers.py`, `layers/RevINs.py`: prediction head, transformer components, and reversible normalization.
- `utils/`: patch and token operations, statistics, metrics, experiment identity, and Reference cache verification.
- `data_provider/`: benchmark data loading and alignment of samples with Reference records.
- `exp/`: training, validation, and evaluation.
- `configs/`: dataset-specific model and training configurations.
- `scripts/ref.py`: Reference construction and verification.
- `scripts/DReSTG_run.sh`, `scripts/DReSTG_cleanup.sh`: experiment execution and per-cell artifact cleanup.

## Reference Mechanism

Reference caches sample-specific statistical relationships used to construct the spatiotemporal graph without recomputing them at every forecasting step.

- `reference/base/<dataset>/<split>/`: `rho.npy` stores signed pairwise Pearson correlations; `stat.npy` stores normalized coarse-window statistics; `idx.npy` maps cached rows to the original samples.
- `reference/st/<dataset>/<split>/`: `pair.npy` stores canonical positive variable-pair candidates selected through cumulative-mass Soft-K; `r.npy` stores their correlation values; `off.npy` indexes the variable-length candidate lists.
- `meta.json` records the cache contract and data configuration; `ok.json` marks verified assets.

The spatiotemporal graph module consumes the verified Reference relationships alongside learned global relations. The cache is constructed per dataset and split and checked for alignment with the input samples.
