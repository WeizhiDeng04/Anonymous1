


import math
import torch
from torch import nn
import torch.nn.functional as F
from layers.GraphPropagation import OneHopMixHopPropagation
from utils.ref_cache import softk_budget


class GlobalGraphModule(nn.Module):
    def __init__(self, d_model, channels, coarse_patch_count, mass_ratio=0.5,
                 alpha=0.05, lambda_init=0.1):
        super().__init__()
        if coarse_patch_count != 7 or d_model % 4 or abs(alpha - .05) > 1e-12 or abs(lambda_init - .1) > 1e-12:
            raise ValueError('DReSTG requires N_C=7, rank D/4, MH1 alpha=0.05, lambda_init=0.1')
        self.channels, self.coarse_patch_count, self.d_model = channels, coarse_patch_count, d_model
        self.rank = d_model // 4
        self.mass_ratio = float(mass_ratio)
        self.mass_budget = softk_budget(channels, mass_ratio)
        with torch.random.fork_rng(devices=[]):
            self.global_mh1 = OneHopMixHopPropagation(
                d_model=d_model, channel_num=channels, coarse_patch_count=1,
                mixhop_depth=1, mixhop_alpha=alpha, support_preserving=False,
                layer_index=0, forecast_effective=True)
            self.relation_projection = nn.Linear(d_model, self.rank, bias=False)
            nn.init.orthogonal_(self.relation_projection.weight, gain=1.0)
        self.theta_lambda_g = nn.Parameter(torch.tensor(math.log(math.expm1(.1)), dtype=torch.float32))
    @property
    def lambda_g(self):
        return F.softplus(self.theta_lambda_g)

    def pool_global_nodes(self, coarse_nodes):
        if coarse_nodes.ndim != 4 or tuple(coarse_nodes.shape[1:]) != (self.channels,7,self.d_model):
            raise ValueError('DReSTG node input must be [B,C,7,D]')
        return coarse_nodes.mean(dim=2)

    def score_global_relations(self, global_nodes):
        
        q = self.relation_projection(global_nodes.float())
        q = F.normalize(q, p=2, dim=-1, eps=1e-8)
        upper = torch.triu_indices(self.channels,self.channels,offset=1,device=global_nodes.device)
        relation_scores = (q[:,upper[0],:] * q[:,upper[1],:]).sum(-1)
        return relation_scores.clamp(-1.,1.), upper

    @staticmethod
    def _selected_prefix(scores, budget):


        if scores.ndim != 2 or not torch.isfinite(scores).all():
            raise ValueError('DReSTG learned score [B,U] must be finite')
        B,U = scores.shape
        if budget < 0:
            raise ValueError('negative mass budget')
        
        order = torch.argsort(scores.detach(),dim=-1,descending=True,stable=True)
        sorted_values = scores.detach().gather(1,order)
        positive = sorted_values > 0
        csum = sorted_values.double().clamp_min(0).cumsum(dim=1)
        included = positive & (csum <= float(budget))
        out = torch.zeros((B,U),device=scores.device,dtype=torch.bool)
        return out.scatter(1,order,included)

    @staticmethod
    def _pair_id(c,d,C):
        return c * (2*C-c-1)//2 + d-c-1

    def _packed_mask(self, idx, B, U):
        if idx.ndim != 2 or idx.shape[-1] != 3:
            raise ValueError('DReSTG pair_index shape [M,3] required')
        mask = torch.zeros((B,U),device=idx.device,dtype=torch.bool)
        if idx.numel():
            bi,c,d = idx.long().unbind(1)
            if torch.any(c>=d) or torch.any(c<0) or torch.any(d>=self.channels) or torch.any(bi<0) or torch.any(bi>=B):
                raise ValueError('DReSTG invalid canonical Reference pair')
            flat = bi*U+self._pair_id(c,d,self.channels)
            if torch.unique(flat).numel()!=flat.numel():
                raise ValueError('DReSTG duplicate Reference parent pair')
            mask[bi,self._pair_id(c,d,self.channels)] = True
        return mask

    @staticmethod
    def _to_packed(mask, upper):
        b, pid = torch.nonzero(mask,as_tuple=True)
        return torch.stack((b,upper[0,pid],upper[1,pid]),dim=1).long(), b,pid

    def forward(self, coarse_nodes, pair_index, pair_r, rho_full):
        global_nodes = self.pool_global_nodes(coarse_nodes)
        B = coarse_nodes.shape[0]
        U = self.channels * (self.channels-1)//2
        if rho_full.ndim != 2 or tuple(rho_full.shape) != (B,U):
            raise ValueError('DReSTG requires canonical rho_full[B,U]')
        rho_full = rho_full.to(coarse_nodes.device,dtype=torch.float32)
        if not torch.isfinite(rho_full).all() or torch.any(rho_full.abs() > 1.0001):
            raise ValueError('DReSTG full Pearson invalid')
        pair_index = pair_index.to(coarse_nodes.device).long()
        pair_r = pair_r.to(coarse_nodes.device,dtype=torch.float32)
        if pair_r.ndim != 1 or pair_r.shape[0] != pair_index.shape[0]:
            raise ValueError('DReSTG positive pair values misaligned')
        scores,upper = self.score_global_relations(global_nodes)
        if scores.shape != (B,U):
            raise ValueError('DReSTG cosine score shape mismatch')
        global_candidate_mask = self._selected_prefix(scores,self.mass_budget)
        st_candidate_mask = self._packed_mask(pair_index,B,U)
        
        if pair_index.numel():
            b0,c0,d0 = pair_index.unbind(1)
            ref_full = rho_full[b0,self._pair_id(c0,d0,self.channels)]
            if not torch.allclose(pair_r,ref_full,atol=1e-6,rtol=1e-6):
                raise ValueError('DReSTG Reference screen.r does not match rho_full')
        union_candidate_mask = global_candidate_mask | st_candidate_mask
        
        
        global_edge_index, bg, pidg = self._to_packed(union_candidate_mask & (scores>0),upper)
        global_edge_weights = scores[bg,pidg].to(dtype=coarse_nodes.dtype)
        global_propagated = self.global_mh1(global_nodes.unsqueeze(2),global_edge_weights.reshape(-1,1,1),global_edge_index)
        self_only_propagated = self.global_mh1(global_nodes.unsqueeze(2),global_edge_weights.new_empty((0,1,1)),global_edge_index.new_empty((0,3)))
        global_relation_delta = (global_propagated-self_only_propagated).squeeze(2)
        global_infusion = self.lambda_g*global_relation_delta
        enhanced_coarse_nodes = coarse_nodes+global_infusion.unsqueeze(2)
        
        st_pair_index,bl,pidl = self._to_packed(union_candidate_mask & (rho_full>0),upper)
        st_pair_rho = rho_full[bl,pidl]
        return enhanced_coarse_nodes, st_pair_index, st_pair_rho
