# -----------------------------------------------------------------------------
# Copyright (c) Yuan Feng et al., 2025
# License: MIT License
# DReSTG clean baseline: DReSTG scientific path only.
# Native PCF, post-SoftK Tiny filtering and mechanism diagnostics are absent.
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatiotemporalRelationEstimator(nn.Module):
    def __init__(self, coarse_patch_count, channel_num, stat_feature_dim,
                 theta_s_init=0.5413248546, boundary_eps=1e-3):
        super(SpatiotemporalRelationEstimator, self).__init__()
        self.coarse_patch_count = int(coarse_patch_count)
        self.channel_num = int(channel_num)
        self.stat_feature_dim = int(stat_feature_dim)
        self.theta_s_init = float(theta_s_init)
        self.boundary_eps = float(boundary_eps)
        if not (0.0 < self.boundary_eps < 1.0):
            raise ValueError('DReSTG boundary_eps must be in (0,1)')
        self.theta_s = nn.Parameter(torch.tensor(self.theta_s_init, dtype=torch.float32))

    @property
    def alpha_s(self):
        return F.softplus(self.theta_s)

    def _fuse_statistical_relations(self, pair_rho, stat_similarity):
        eps = self.boundary_eps
        rho_clamped = torch.clamp(pair_rho, max=1.0 - eps)
        s_stat_clamped = torch.clamp(stat_similarity, min=-1.0 + eps, max=1.0 - eps)
        z_rho = torch.atanh(rho_clamped).view(-1, 1, 1)
        z_stat = torch.atanh(s_stat_clamped)
        delta_stat = self.alpha_s * z_stat
        z_fusion = z_rho + delta_stat
        return torch.tanh(z_fusion)

    def forward(self, x_in, pair_index, pair_rho, packed=True):
        if not packed:
            raise ValueError('DReSTG only accepts packed/ragged SoftK relation batches')
        self._validate_pair_rho(pair_index, pair_rho)
        stat_similarity = self.compute_statistical_similarity(x_in, pair_index)
        if stat_similarity.shape[0] == 0:
            return stat_similarity
        st_relation_scores = self._fuse_statistical_relations(pair_rho, stat_similarity)
        return F.relu(st_relation_scores)

    def _validate_pair_rho(self, pair_index, pair_rho):
        if pair_index.ndim != 2 or pair_index.shape[-1] != 3:
            raise ValueError(f'packed pair_index must be [M,3], got {tuple(pair_index.shape)}')
        if pair_rho.ndim != 1 or int(pair_rho.shape[0]) != int(pair_index.shape[0]):
            raise ValueError('DReSTG pair_r must be [M] aligned exactly with pair_index')
        if pair_rho.numel():
            if not torch.isfinite(pair_rho).all():
                raise ValueError('DReSTG pair_r contains non-finite values')
            if torch.any(pair_rho <= 0) or torch.any(pair_rho > 1):
                raise ValueError('DReSTG pair_r must be selected raw positive Pearson in (0,1]')

    def compute_statistical_similarity(self, x_in, pair_index):
        if pair_index.ndim != 2 or pair_index.shape[-1] != 3:
            raise ValueError(f'packed pair_index must be [M,3], got {tuple(pair_index.shape)}')
        if pair_index.numel() == 0:
            return x_in.new_empty((0, self.coarse_patch_count, self.coarse_patch_count))
        batch = pair_index[:, 0].long()
        src = pair_index[:, 1].long()
        dst = pair_index[:, 2].long()
        if torch.any(src >= dst):
            raise ValueError('DReSTG packed relation pairs must be canonical unique-undirected c<d')
        flat_key = batch * (self.channel_num * self.channel_num) + src * self.channel_num + dst
        skey, _ = torch.sort(flat_key)
        if skey.numel() > 1 and torch.any(skey[1:] == skey[:-1]):
            raise ValueError('DReSTG contains duplicate undirected pair within a sample')
        s = x_in[batch, src]
        d = x_in[batch, dst]
        s = s.unsqueeze(2).expand(-1, -1, self.coarse_patch_count, -1)
        d = d.unsqueeze(1).expand(-1, self.coarse_patch_count, -1, -1)
        return torch.cosine_similarity(s, d, dim=-1)
