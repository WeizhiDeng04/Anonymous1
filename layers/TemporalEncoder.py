
# Copyright (c) Yuan Feng et al., 2025
# License: MIT License


from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.PatchTST_layers import positional_encoding, Transpose, get_activation_fn
from utils.token_utils import pool_coarse_patch_nodes, interleave_coarse_fine_tokens, split_coarse_fine_tokens
from layers.GraphPropagation import OneHopMixHopPropagation
from layers.DualResolutionGraph import GlobalGraphModule

class TSTiEncoder(nn.Module):
    def __init__(self, c_in, interior_fine_patch_count, fine_patch_len,
                    coarse_patch_count, fine_patches_per_coarse, gnn_r_degree, gnn_hidden_size,
                    mixhop_depth=1, mixhop_alpha=0.05, mixhop_support_preserving=False,
                    lambda_init=0.1, softk_mass_ratio=0.5,
                    max_seq_len=1024, n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                    d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu",
                    key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                    pe='zeros', learn_pe=True, verbose=False, **kwargs):
        
        super().__init__()
        
        self.interior_fine_patch_count = interior_fine_patch_count
        self.fine_patch_len = fine_patch_len
        self.coarse_patch_count = coarse_patch_count
        self.fine_patches_per_coarse = fine_patches_per_coarse

        token_count = interior_fine_patch_count + (coarse_patch_count - 1) + coarse_patch_count
        
        self.fine_patch_embedding = nn.Linear(fine_patch_len, d_model)
        self.token_count = token_count
        
        self.position_embedding = positional_encoding(pe, learn_pe, token_count, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        self.encoder = TSTEncoder(token_count, d_model, n_heads, c_in=c_in,
                                    coarse_patch_count=coarse_patch_count, fine_patches_per_coarse=fine_patches_per_coarse, gnn_r_degree=gnn_r_degree, gnn_hidden_size=gnn_hidden_size,
                                    mixhop_depth=mixhop_depth, mixhop_alpha=mixhop_alpha, mixhop_support_preserving=mixhop_support_preserving,
                                    lambda_init=lambda_init, softk_mass_ratio=softk_mass_ratio,
                                    d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                    pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers)
    
    def forward(self, x, pair_index, pair_r, rho_full, statistics, st_relation_estimator) -> Tensor:
        n_vars = x.size(1)
        fine_tokens = self.fine_patch_embedding(x)
        coarse_nodes = pool_coarse_patch_nodes(fine_tokens, self.coarse_patch_count, self.fine_patches_per_coarse)
        tokens = interleave_coarse_fine_tokens(coarse_nodes, fine_tokens, self.coarse_patch_count, self.fine_patches_per_coarse)
        u = tokens.reshape(tokens.size(0)*n_vars, tokens.size(-2), tokens.size(-1))
        u = self.dropout(u + self.position_embedding)
        u = u.reshape(-1, n_vars, u.size(-2), u.size(-1))
        z = self.encoder(u, pair_index, pair_r, rho_full, statistics, st_relation_estimator)
        coarse_nodes, fine_tokens = split_coarse_fine_tokens(z, self.coarse_patch_count, self.fine_patches_per_coarse)
        return coarse_nodes, fine_tokens


class TSTEncoder(nn.Module):
    def __init__(self, token_count, d_model, n_heads, c_in,
                    coarse_patch_count, fine_patches_per_coarse, gnn_r_degree, gnn_hidden_size,
                    mixhop_depth=1, mixhop_alpha=0.05, mixhop_support_preserving=False,
                    lambda_init=0.1, softk_mass_ratio=0.5,
                    d_k=None, d_v=None, d_ff=None, 
                    norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                    res_attention=False, n_layers=2, pre_norm=False):
        super().__init__()
        
        assert n_layers >= 1, "please give at least one layer"

        self.main_trans_layers = nn.ModuleList([TSTEncoderLayer(token_count, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                        attn_dropout=attn_dropout, dropout=dropout,
                                                        activation=activation, res_attention=res_attention,
                                                        pre_norm=pre_norm) for i in range(n_layers)])
        
        
        
        
        
        self.st_graph_layers = nn.ModuleList([
            OneHopMixHopPropagation(
                d_model=d_model, channel_num=c_in, coarse_patch_count=coarse_patch_count,
                mixhop_depth=mixhop_depth, mixhop_alpha=mixhop_alpha,
                support_preserving=mixhop_support_preserving, layer_index=k,
                forecast_effective=(k < n_layers - 1),
            )
            for k in range(n_layers)
        ])
        
        
        self.global_graph = GlobalGraphModule(
            d_model=d_model, channels=c_in, coarse_patch_count=coarse_patch_count,
            mass_ratio=softk_mass_ratio,
            alpha=mixhop_alpha, lambda_init=lambda_init)

        self.c_in = c_in
        self.res_attention = res_attention
        self.coarse_patch_count = coarse_patch_count
        self.fine_patches_per_coarse = fine_patches_per_coarse
        
    def forward(self, src:Tensor, pair_index:Tensor, pair_r:Tensor, rho_full:Tensor, statistics:Tensor, st_relation_estimator, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        
        output = src
        scores = None
        local_ids, local_weights = None, None
        
        if self.res_attention:
            for layer_id, (main_trans_mods, segment_graph_mod) in enumerate(zip(self.main_trans_layers, self.st_graph_layers)):
                
                output = output.flatten(start_dim=0, end_dim=1)
                
                output, scores = main_trans_mods(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
                
                output = output.reshape(-1, self.c_in, output.shape[-2], output.shape[-1])
                
                coarse_nodes, fine_tokens = split_coarse_fine_tokens(output, self.coarse_patch_count, self.fine_patches_per_coarse)
                if layer_id == 0:
                    coarse_nodes, local_ids, local_r = self.global_graph(coarse_nodes, pair_index, pair_r, rho_full)
                    
                    
                    local_weights = st_relation_estimator(statistics, local_ids, local_r, packed=True)
                coarse_nodes = segment_graph_mod(coarse_nodes, local_weights, local_ids)
                output = interleave_coarse_fine_tokens(coarse_nodes, fine_tokens, self.coarse_patch_count, self.fine_patches_per_coarse)
            
            return output
        else:
            raise ValueError("Residual attention is required by frozen DReSTG")


class TSTEncoderLayer(nn.Module):
    def __init__(self, token_count, d_model, n_heads, d_k=None, d_v=None, d_ff=256,
                    norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)
        
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))
        
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)
        
        self.pre_norm = pre_norm

    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:
        
        if self.pre_norm:
            src = self.norm_attn(src)
        
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        
        src = src + self.dropout_attn(src2)
        if not self.pre_norm:
            src = self.norm_attn(src)
        
        if self.pre_norm:
            src = self.norm_ffn(src)
        
        src2 = self.ff(src)
        
        src = src + self.dropout_ffn(src2)
        if not self.pre_norm:
            src = self.norm_ffn(src)
        
        if self.res_attention:
            return src, scores
        else:
            return src


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        super().__init__()
        
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
        
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)
        
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)
        
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))
    
    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        
        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q
        
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)
        
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        output = self.to_out(output)
        
        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        
        attn_scores = torch.matmul(q, k) * self.scale
        
        if prev is not None: attn_scores = attn_scores + prev
        
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask
        
        if key_padding_mask is not None:
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        
        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights
