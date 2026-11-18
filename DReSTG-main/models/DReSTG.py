# -----------------------------------------------------------------------------
# Copyright (c) Yuan Feng et al., 2025
# License: MIT License
# Description: DReSTG entry point; single-path DReSTG.
# -----------------------------------------------------------------------------

__all__ = ['DReSTG']

from typing import Optional
from torch import nn, Tensor

from layers.DReSTG_backbone import DReSTGBackbone


class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None,
                 d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0.,
                 act:str='gelu', key_padding_mask:bool='auto', padding_var:Optional[int]=None,
                 attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False,
                 store_attn:bool=False, pe:str='zeros', learn_pe:bool=True,
                 pretrain_head:bool=False, head_type='flatten', verbose:bool=False, **kwargs):
        super().__init__()

        c_in = configs.enc_in
        device = configs.device
        eps = configs.eps
        context_window = configs.seq_len
        target_window = configs.pred_len
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        individual = configs.individual
        fine_patch_len = configs.fine_patch_len
        fine_patch_stride = configs.fine_patch_stride
        padding_patch = configs.padding_patch
        revin = configs.revins
        affine = configs.affine
        subtract_last = configs.subtract_last
        decomposition = configs.decomposition
        decomposition_kernel_size = configs.decomposition_kernel_size
        coarse_patch_count = configs.coarse_patch_count
        gnn_r_degree = configs.gnn_r_degree
        gnn_hidden_size = configs.gnn_hidden_size
        mixhop_depth = configs.mixhop_depth
        mixhop_alpha = configs.mixhop_alpha
        mixhop_support_preserving = configs.mixhop_support_preserving
        theta_s_init = configs.theta_s_init
        boundary_eps = configs.boundary_eps
        lambda_init = configs.lambda_init
        softk_mass_ratio = configs.softk_mass_ratio

        if decomposition:
            raise ValueError('DReSTG requires decomposition=False')
        common = dict(
            c_in=c_in, device=device, eps=eps,
            coarse_patch_count=coarse_patch_count, gnn_r_degree=gnn_r_degree,
            gnn_hidden_size=gnn_hidden_size, mixhop_depth=mixhop_depth,
            mixhop_alpha=mixhop_alpha, mixhop_support_preserving=mixhop_support_preserving,
            theta_s_init=theta_s_init, boundary_eps=boundary_eps,
            lambda_init=lambda_init, softk_mass_ratio=softk_mass_ratio,
            context_window=context_window, target_window=target_window,
            fine_patch_len=fine_patch_len, fine_patch_stride=fine_patch_stride, max_seq_len=max_seq_len,
            n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v,
            d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
            act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
            attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
            pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout,
            head_dropout=head_dropout, padding_patch=padding_patch,
            pretrain_head=pretrain_head, head_type=head_type, individual=individual,
            revin=revin, affine=affine, subtract_last=subtract_last, verbose=verbose, **kwargs
        )

        self.model = DReSTGBackbone(**common)

    def forward(self, x, relation_cache=None):
        x = x.permute(0, 2, 1)
        out_y = self.model(x, relation_cache=relation_cache)
        return out_y.permute(0, 2, 1)
