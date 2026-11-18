"""Single-model DReSTG recipe and identity (dual-resolution union)."""
from __future__ import annotations
from typing import Any, Dict

RECIPE_SCHEMA = 'DReSTG_recipe_v1'
IDENTITY_SCHEMA = 'DReSTG_cell_identity_v1'
SEMANTIC_FINGERPRINT = 'DReSTG_GLOBAL_FINE_P336_D4_LEARNED_COSINE_SOFTK_MH1_ONCE_RELATION_ONLY_DELTA_LEARNABLE_SOFTPLUS_0P1_P48_PRIOR_GUIDED_ALLFINE_DECODER'
SCREEN_SEMANTICS = 'drestg_raw_positive_undirected_cumulative_mass_softk_packed_no_refill'
FUSION_SEMANTICS = (
    'rho_bar=min(rho,1-eps);S_bar=clip(S_stat,-1+eps,1-eps);'
    'z_rho=atanh(rho_bar);z_stat=atanh(S_bar);'
    'delta=softplus(theta)*z_stat;z_fusion=z_rho+delta;'
    'R_init=tanh(z_fusion);A_undir=ReLU(R_init);NO_PCF;NO_TINY_FILTER'
)
RELATION_SETTLEMENT = 'relu_on_prior_guided_relation'


def semantic_fingerprint() -> str:
    return SEMANTIC_FINGERPRINT


def screen_semantics() -> str:
    return SCREEN_SEMANTICS


def fusion_semantics() -> str:
    return FUSION_SEMANTICS


def relation_settlement() -> str:
    return RELATION_SETTLEMENT


def _plain(value: Any):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def recipe_from_args(args) -> Dict[str, Any]:
    fields = [
        'model', 'data', 'root_path', 'data_path', 'target', 'scale',
        'seq_len', 'label_len', 'pred_len', 'random_seed',
        'enc_in', 'fine_patch_len', 'padding_patch', 'fine_patch_stride', 'coarse_patch_count',
        'd_model', 'n_heads', 'e_layers', 'd_layers', 'd_ff',
        'gnn_r_degree', 'gnn_hidden_size',
        'revins', 'affine', 'subtract_last', 'decomposition', 'decomposition_kernel_size',
        'use_amp', 'itr', 'learning_rate', 'patience', 'train_epochs',
        'pct_start', 'lradj', 'dropout', 'fc_dropout', 'head_dropout', 'individual',
        'batch_size', 'num_workers', 'eps', 'softk_mass_ratio',
        'mixhop_depth', 'mixhop_alpha', 'mixhop_support_preserving',
        'theta_s_init', 'boundary_eps', 'lambda_init', 
        'drestg_dataset_id', 'drestg_reference_root', 'drestg_reference_mode', 'config_yaml',
    ]
    out = {
        'schema': RECIPE_SCHEMA,
        'relation_settlement': RELATION_SETTLEMENT,
        'fine_patching': 'continuous_global_unfold_with_seam_recovered',
        'coarse_pool': 'fine_neighborhood_mean',
        'global_graph': 'P336_D4_bilinear_cosine_positive_dynamic_softk_MH1_once',
        'global_infusion': 'softplus_lambda_0p1_uniform_relation_only_delta',
        'coarse_node_builder': 'mean_pool_7_to_1',
        'support_collaboration': 'union_only_native_weighting',
        'decoder': 'all_fine_tokens',
    }
    for key in fields:
        if not hasattr(args, key):
            raise ValueError('missing recipe argument %s' % key)
        out[key] = _plain(getattr(args, key))
    return out


def cell_identity_from_args(args) -> Dict[str, Any]:
    dataset_id = str(getattr(args, 'drestg_dataset_id', '') or getattr(args, 'data', ''))
    return {
        'schema': IDENTITY_SCHEMA,
        'dataset': dataset_id,
        'horizon': int(args.pred_len),
        'seed': int(args.random_seed),
        'model_id': str(args.model_id),
        'fine_patching': 'continuous_global_unfold_with_seam_recovered',
        'coarse_pool': 'fine_neighborhood_mean',
        'coarse_node_builder': 'mean_pool_7_to_1',
        'support_collaboration': 'union_only_native_weighting',
        'decoder': 'all_fine_tokens',
        'global_metric_rank': int(args.d_model)//4,
        'metric_init': 'semi_orthogonal_gain1',
        'lambda_init': float(args.lambda_init),
        'global_graph': 'P336_D4_bilinear_cosine_positive_dynamic_softk_MH1_once',
        'global_infusion': 'uniform_relation_only_delta',
        'theta_s_init': float(args.theta_s_init),
        'boundary_eps': float(args.boundary_eps),
        'reference_mode': str(args.drestg_reference_mode),
        'reference_root': str(args.drestg_reference_root),
        'semantic_fingerprint': semantic_fingerprint(),
        'screen_semantics': screen_semantics(),
        'fusion_semantics': fusion_semantics(),
        'relation_settlement': relation_settlement(),
        'propagation': 'MH1',
    }
