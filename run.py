


import argparse
import json
import random
from pathlib import Path

import yaml

from utils.drestg_identity import cell_identity_from_args, recipe_from_args

THETA_S_INIT = 0.5413248546
BOUNDARY_EPS = 1e-3
DEFAULT_REFERENCE_ROOT = 'reference'


def build_setting(args, ii):
    return '{}_dm{}_nh{}_el{}_dl{}_df{}_nc{}_km{}_union_prior_relu_mh1_a{}_{}'.format(
        args.model_id, args.d_model, args.n_heads, args.e_layers, args.d_layers,
        args.d_ff, args.coarse_patch_count, args.softk_mass_ratio,
        str(args.mixhop_alpha).replace('.', 'p'), ii,
    )


if __name__ == '__main__':
    config_parser = argparse.ArgumentParser(description='config parser', add_help=False)
    config_parser.add_argument('-c', '--config_yaml', default='train.yml', type=str, metavar='FILE')
    config_args, remaining_args = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(description='DReSTG dual-resolution learned-global/statistical-ST Union forecasting')
    parser.add_argument('--random_seed', type=int, default=2021)
    parser.add_argument('--data', type=str, required=True, default='ETTh1')
    parser.add_argument('--root_path', type=str, default='datasets/Ett_Series')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--scale', type=bool, default=True)
    parser.add_argument('--seq_len', type=int, default=336)
    parser.add_argument('--label_len', type=int, default=0)
    parser.add_argument('--pred_len', type=int, default=720)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', type=bool, default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3')
    parser.add_argument('--model_id', type=str, required=True, default='test')
    parser.add_argument('--model', type=str, required=True, default='DReSTG')
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--fine_patch_len', type=int, default=16)
    parser.add_argument('--padding_patch', default='start')
    parser.add_argument('--fine_patch_stride', type=int, default=8)
    parser.add_argument('--coarse_patch_count', type=int, default=8)
    parser.add_argument('--gnn_r_degree', type=int, default=1)
    parser.add_argument('--gnn_hidden_size', type=int, default=32)
    parser.add_argument('--revins', type=bool, default=True)
    parser.add_argument('--affine', type=bool, default=False)
    parser.add_argument('--subtract_last', type=bool, default=False)
    parser.add_argument('--decomposition', type=bool, default=False)
    parser.add_argument('--decomposition_kernel_size', type=int, default=25)
    parser.add_argument('--is_training', type=int, required=True, default=1)
    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--train_epochs', type=int, default=100)
    parser.add_argument('--pct_start', type=float, default=0.3)
    parser.add_argument('--lradj', type=str, default='type3')
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--fc_dropout', type=float, default=0.05)
    parser.add_argument('--head_dropout', type=float, default=0.0)
    parser.add_argument('--individual', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--eps', type=float, default=0.00001)
    parser.add_argument('--softk_mass_ratio', type=float, required=False, default=None)

    parser.add_argument('--mixhop_depth', type=int, default=1, choices=[1])
    parser.add_argument('--mixhop_alpha', type=float, default=0.05)
    parser.add_argument('--mixhop_support_preserving', action='store_true', default=False)
    parser.add_argument('--theta_s_init', type=float, default=THETA_S_INIT)
    parser.add_argument('--boundary_eps', type=float, default=BOUNDARY_EPS)
    parser.add_argument('--lambda_init', type=float, default=0.1)

    
    parser.add_argument('--drestg_dataset_id', type=str, default='')
    parser.add_argument('--drestg_reference_root', type=str, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument('--drestg_reference_mode', type=str, default='reuse', choices=['build', 'reuse'])
    parser.add_argument('--drestg_recipe_out', type=str, default='')

    with open(config_args.config_yaml, 'r', encoding='utf-8') as f:
        default_arg = yaml.safe_load(f)
        if not isinstance(default_arg, dict):
            raise ValueError('config YAML must contain a mapping')
        parser.set_defaults(**default_arg)
    args, unknown_args = parser.parse_known_args(remaining_args)
    if unknown_args:
        raise ValueError('unknown DReSTG command-line arguments: %s' % unknown_args)
    args.config_yaml = config_args.config_yaml

    if args.model != 'DReSTG':
        raise ValueError('DReSTG only supports model=DReSTG')
    if args.softk_mass_ratio is None:
        raise ValueError('DReSTG requires frozen softk_mass_ratio from config')
    if bool(args.decomposition) or bool(args.affine) or bool(args.subtract_last):
        raise ValueError('DReSTG Reference contract requires decomposition=False, affine=False, subtract_last=False')
    if int(args.gnn_r_degree) != 1:
        raise ValueError('DReSTG preserves gnn_r_degree=1')
    if int(args.mixhop_depth) != 1 or bool(args.mixhop_support_preserving):
        raise ValueError('DReSTG freezes MH1 only; MH2/SP-MH2 are forbidden')
    if abs(float(args.mixhop_alpha) - 0.05) > 1e-12:
        raise ValueError('DReSTG freezes mixhop_alpha=0.05')
    if abs(float(args.theta_s_init) - THETA_S_INIT) > 1e-10:
        raise ValueError('DReSTG theta_s_init is frozen to 0.5413248546 (alpha_init=1)')
    if abs(float(args.boundary_eps) - BOUNDARY_EPS) > 1e-15:
        raise ValueError('DReSTG boundary_eps is frozen to 1e-3')
    if abs(float(args.lambda_init) - 0.1) > 1e-12:
        raise ValueError('DReSTG requires lambda_init=0.1')
    if int(args.coarse_patch_count)!=7:
        raise ValueError('DReSTG requires exactly seven P48 heads')
    if not str(args.drestg_reference_root):
        raise ValueError('DReSTG requires an explicit non-empty Reference root')

    if str(args.drestg_recipe_out):
        out = Path(args.drestg_recipe_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {'cell_identity': cell_identity_from_args(args), 'recipe': recipe_from_args(args)}
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
        print('[DReSTG RECIPE] wrote', out)
        raise SystemExit(0)

    import numpy as np
    import torch
    import dgl
    from exp.exp_main import Exp_Main

    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    dgl.random.seed(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        args.device_ids = [int(i) for i in args.devices.split(',')]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)
    print('DReSTG: continuous fine grid + Global MH1 + prior-guided ST MH1 + MeanPool7 + Union')

    Exp = Exp_Main
    if args.is_training:
        for ii in range(args.itr):
            setting = build_setting(args, ii)
            exp = Exp(args)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = build_setting(args, ii)
        exp = Exp(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
