
# Copyright (c) Yuan Feng et al., 2025
# License: MIT License


import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn


class dgl_graph_layer_for_trans(nn.Module):
    def __init__(self, d_model, hidden_size, n_layer, method_type, aggregator_type='gcn', feat_drop=0., norm=None, activation=None):
        super(dgl_graph_layer_for_trans, self).__init__()
        
        self.d_model = d_model
        
        if method_type == 'SAGE':
            if n_layer == 1:
                self.graph_layers_list = nn.ModuleList([dglnn.SAGEConv(d_model, d_model, aggregator_type=aggregator_type, feat_drop=feat_drop, norm=norm, activation=activation)])
                
            elif n_layer == 2:
                self.graph_layers_list = nn.ModuleList([dglnn.SAGEConv(d_model, hidden_size, aggregator_type=aggregator_type, feat_drop=feat_drop, norm=norm, activation=activation),
                                                        dglnn.SAGEConv(hidden_size, d_model, aggregator_type=aggregator_type, feat_drop=feat_drop, norm=norm, activation=activation)
                                                        ])
            else:
                hidden_layer_num =  n_layer - 2
                rough_layers = [dglnn.SAGEConv(d_model, hidden_size, aggregator_type=aggregator_type, feat_drop=feat_drop, norm=norm, activation=activation)]
                rough_layers.extend([dglnn.SAGEConv(hidden_size, hidden_size, aggregator_type=aggregator_type, feat_drop=feat_drop, norm=norm, activation=activation) for i in range(hidden_layer_num)])
                rough_layers.append(dglnn.SAGEConv(hidden_size, d_model, aggregator_type=aggregator_type, feat_drop=feat_drop, norm=norm, activation=activation))
                self.graph_layers_list = nn.ModuleList(rough_layers)
        else:
            raise ValueError("do not have such gnn method type")
    
    def forward(self, g, x_in, weights):
        
        node_num = x_in.shape[-2]
        x_flatten = torch.flatten(x_in, start_dim=0, end_dim=-2)
        
        for graph_layer in self.graph_layers_list:
            x_flatten =  graph_layer(g, x_flatten, weights)
        
        x_out = x_flatten.reshape(-1, node_num, self.d_model)
        
        return x_out


class dgl_graph_for_trans(nn.Module):
    def __init__(self,
                    d_model,
                    graph_r,
                    graph_hidden_size=64,
                    graph_method_type='SAGE'):
        super(dgl_graph_for_trans, self).__init__()
        
        self.graph_layer = dgl_graph_layer_for_trans(d_model, graph_hidden_size, graph_r, method_type=graph_method_type)
        
    def forward(self, x_in, g, weights):
        
        temp_shape = x_in.shape
        x_out = torch.flatten(x_in, start_dim=1, end_dim=2)
        x_out = self.graph_layer(g, x_out, weights)
        x_out = x_out.reshape(temp_shape)

        return x_out

import torch.nn.functional as F

class OneHopMixHopPropagation(nn.Module):


    def __init__(self, d_model, channel_num, coarse_patch_count, mixhop_depth=1,
                 mixhop_alpha=0.05, support_preserving=False,
                 layer_index=0, forecast_effective=True):
        super(OneHopMixHopPropagation, self).__init__()
        if int(mixhop_depth) != 1:
            raise ValueError('DReSTG freezes MH1 only: mixhop_depth must be 1')
        if bool(support_preserving):
            raise ValueError('DReSTG has no SP-MH2 path')
        self.d_model = int(d_model)
        self.channel_num = int(channel_num)
        self.coarse_patch_count = int(coarse_patch_count)
        self.mixhop_depth = 1
        self.mixhop_alpha = float(mixhop_alpha)
        self.layer_index = int(layer_index)
        self.forecast_effective = bool(forecast_effective)
        self.hop_projection = nn.Linear(2 * self.d_model, self.d_model, bias=True)

    def _validate_packed(self, pair_weights, pair_index, batch_size):
        if pair_weights.ndim != 3 or pair_index.ndim != 2:
            raise ValueError('DReSTG MH1 expects pair_weights [M,L,L] and pair_index [M,3]')
        if pair_index.shape != (pair_weights.shape[0], 3):
            raise ValueError('DReSTG pair_index shape mismatch')
        if pair_weights.shape[-2:] != (self.coarse_patch_count, self.coarse_patch_count):
            raise ValueError('DReSTG coarse-node shape mismatch')
        if pair_index.numel():
            b = pair_index[:, 0].long()
            if torch.any(b < 0) or torch.any(b >= batch_size):
                raise ValueError('DReSTG batch id out of range')
        return pair_weights, pair_index, batch_size

    def _apply_undirected(self, pair_weights, pair_index, x):

        bsz, channels, l, d = x.shape
        if channels != self.channel_num or l != self.coarse_patch_count:
            raise ValueError('DReSTG MH1 node representation shape mismatch')
        w, idx, _ = self._validate_packed(pair_weights, pair_index, bsz)
        if w.numel() == 0:
            return x, torch.ones((bsz, channels, l), dtype=x.dtype, device=x.device)
        if torch.any(w < 0):
            raise ValueError('DReSTG expects non-negative final relation edge weights')
        batch = idx[:, 0].long(); c = idx[:, 1].long(); dch = idx[:, 2].long()
        if torch.any(c >= dch):
            raise ValueError('DReSTG relation pairs must be unique-undirected canonical c<d')
        key = batch * (self.channel_num * self.channel_num) + c * self.channel_num + dch
        skey, _ = torch.sort(key)
        if skey.numel() > 1 and torch.any(skey[1:] == skey[:-1]):
            raise ValueError('duplicate undirected relation pair at MH1 boundary')

        blocks = torch.cat([w.transpose(-1, -2), w], dim=0)
        msg_batch = torch.cat([batch, batch], dim=0)
        src = torch.cat([c, dch], dim=0)
        dst = torch.cat([dch, c], dim=0)

        flat_nodes = bsz * channels
        flat_src = msg_batch * channels + src
        flat_dst = msg_batch * channels + dst
        degree = torch.ones((flat_nodes, l), dtype=x.dtype, device=x.device)
        degree.index_add_(0, flat_dst, blocks.sum(dim=-1))
        norm_blocks = blocks / degree.index_select(0, flat_dst).unsqueeze(-1).clamp_min(1e-12)

        xflat = x.reshape(flat_nodes, l, d)
        src_x = xflat.index_select(0, flat_src)
        msg = torch.matmul(norm_blocks, src_x)
        out = torch.zeros_like(xflat)
        out.index_add_(0, flat_dst, msg)
        out = out + xflat / degree.unsqueeze(-1).clamp_min(1e-12)
        return out.reshape(bsz, channels, l, d), degree.reshape(bsz, channels, l)

    def forward(self, x_in, pair_weights, pair_index):
        h0 = x_in
        ph0, degree = self._apply_undirected(pair_weights, pair_index, h0)
        h1 = self.mixhop_alpha * h0 + (1.0 - self.mixhop_alpha) * ph0
        cat = torch.cat([h0, h1], dim=-1)
        out = self.hop_projection(cat)
        return out
