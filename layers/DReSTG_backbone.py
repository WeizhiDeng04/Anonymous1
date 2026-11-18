
# Copyright (c) Yuan Feng et al., 2025
# License: MIT License


__all__ = ['DReSTGBackbone']

from typing import Optional
import torch
from torch import nn, Tensor

from layers.RevINs import RevIN
from layers.SpatiotemporalRelation import SpatiotemporalRelationEstimator
from layers.ForecastDecoder import Flatten_Head
from layers.TemporalEncoder import TSTiEncoder
from utils.patch_utils import compute_patch_layout, extract_fine_patches


class DReSTGBackbone(nn.Module):
    def __init__(self, c_in:int, device: str, eps: float,
                 coarse_patch_count:int, gnn_r_degree:int, gnn_hidden_size:int,
                 context_window:int, target_window:int, fine_patch_len:int, fine_patch_stride:int,
                 mixhop_depth:int=1, mixhop_alpha:float=0.05, mixhop_support_preserving:bool=False,
                 theta_s_init:float=0.5413248546, boundary_eps:float=1e-3,
                 lambda_init:float=0.1, softk_mass_ratio:float=0.5,
                 max_seq_len:Optional[int]=1024,
                 n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str='gelu', key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout=0, padding_patch=None,
                 pretrain_head:bool=False, head_type='flatten', individual=False, revin=True, affine=True, subtract_last=False,
                 verbose:bool=False, **kwargs):
        super(DReSTGBackbone, self).__init__()

        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        self.device = device
        self.statistics_num = 2
        self.subtract_last = subtract_last
        self.eps = eps
        self.c_in = c_in
        self.fine_patch_len = fine_patch_len
        self.fine_patch_stride = fine_patch_stride
        self.padding_patch = padding_patch
        self.coarse_patch_count = coarse_patch_count

        interior_fine_patch_count, fine_patches_per_coarse, x_padding_len = compute_patch_layout(
            seq_len=context_window, coarse_patch_count=coarse_patch_count, fine_patch_len=fine_patch_len, fine_patch_stride=fine_patch_stride
        )
        self.coarse_patch_len = fine_patch_len + (fine_patches_per_coarse - 1) * fine_patch_stride
        self.x_padding_len = x_padding_len

        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, x_padding_len))
        elif padding_patch == 'start':
            self.padding_patch_layer = nn.ReplicationPad1d((x_padding_len, 0))
        else:
            raise ValueError('Invalid padding patch type')

        self.backbone = TSTiEncoder(
            c_in=c_in, interior_fine_patch_count=interior_fine_patch_count, fine_patch_len=fine_patch_len,
            coarse_patch_count=coarse_patch_count, fine_patches_per_coarse=fine_patches_per_coarse,
            gnn_r_degree=gnn_r_degree, gnn_hidden_size=gnn_hidden_size,
            mixhop_depth=mixhop_depth, mixhop_alpha=mixhop_alpha,
            mixhop_support_preserving=mixhop_support_preserving,
            lambda_init=lambda_init, softk_mass_ratio=softk_mass_ratio,
            max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model, n_heads=n_heads,
            d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
            dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
            attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
            pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs
        )

        
        self.statistic_norm = nn.LayerNorm(normalized_shape=[c_in, coarse_patch_count], elementwise_affine=affine)
        self.st_relation_estimator = SpatiotemporalRelationEstimator(
            coarse_patch_count=coarse_patch_count, channel_num=c_in, stat_feature_dim=self.statistics_num,
            theta_s_init=theta_s_init, boundary_eps=boundary_eps,
        )

        self.fine_flatten_dim = (interior_fine_patch_count + coarse_patch_count - 1) * d_model
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual
        if self.pretrain_head:
            self.forecast_decoder = self.create_pretrain_head(self.fine_flatten_dim, c_in, fc_dropout)
        elif head_type == 'flatten':
            self.forecast_decoder = Flatten_Head(
                self.individual, self.c_in, self.fine_flatten_dim,
                target_window, head_dropout=head_dropout
            )

    def forward(self, z:Tensor, relation_cache=None):
        if relation_cache is None:
            raise RuntimeError('DReSTG requires project-local verified native Reference payload')
        kind = relation_cache.get('cache_kind')
        if kind != 'DRESTG-DUAL-RESOLUTION-V1':
            raise ValueError(f'unexpected DReSTG cache kind: {kind!r}')
        required = ('pair_index', 'pair_r', 'offsets', 'statistics', 'packed', 'rho_full')
        missing = [k for k in required if k not in relation_cache]
        if missing:
            raise ValueError(f'DReSTG payload missing {missing}')

        pair_index = relation_cache['pair_index'].to(z.device).long()
        pair_r = relation_cache['pair_r'].to(z.device, dtype=z.dtype)
        offsets = relation_cache['offsets'].to(z.device).long()
        coarse_statistics = relation_cache['statistics'].to(z.device, dtype=z.dtype)
        packed = bool(relation_cache['packed'])
        rho_full = relation_cache['rho_full'].to(z.device,dtype=torch.float32)
        if coarse_statistics.ndim != 4 or coarse_statistics.shape[1:] != (self.c_in, self.coarse_patch_count, 2):
            raise ValueError(
                f'DReSTG statistics shape mismatch: {tuple(coarse_statistics.shape)}; '
                f'expected [B,{self.c_in},{self.coarse_patch_count},2]'
            )
        if not packed:
            raise ValueError('DReSTG must use packed/ragged relation batches')
        if pair_r.ndim != 1 or pair_r.shape[0] != pair_index.shape[0]:
            raise ValueError('DReSTG pair_r must align one-to-one with pair_index')
        if offsets.ndim != 1 or offsets.shape[0] != coarse_statistics.shape[0] + 1:
            raise ValueError('DReSTG offsets must be [B+1]')
        if int(offsets[0].item()) != 0 or int(offsets[-1].item()) != int(pair_index.shape[0]):
            raise ValueError('DReSTG offsets do not cover packed relation array')

        z = self.padding_patch_layer(z)
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0, 2, 1)

        
        z = extract_fine_patches(z, self.coarse_patch_count, self.fine_patch_len, self.fine_patch_stride)
        coarse_output, z_pred = self.backbone(z, pair_index, pair_r, rho_full, coarse_statistics, self.st_relation_estimator)

        z_pred = z_pred.permute(0, 1, 3, 2)
        z_pred = self.forecast_decoder(z_pred)
        if self.revin:
            z_pred = z_pred.permute(0, 2, 1)
            z_pred = self.revin_layer(z_pred, 'denorm')
            z_pred = z_pred.permute(0, 2, 1)
        return z_pred

    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout), nn.Conv1d(head_nf, vars, 1))
