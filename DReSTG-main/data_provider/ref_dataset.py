"""Dataset wrapper for the clean DReSTG model using verified DReSTG native Reference assets.

Scientific contract:
- Reference screen is already positive-only cumulative-mass SoftK;
- runtime consumes the verified screen unchanged;
- no post-Screen tiny filter, no refill, no padding;
- packed/ragged unique-undirected representation is preserved.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from utils.ref_cache import load_base, load_screen, screen_payload


class DReSTGReferenceDataset(Dataset):
    def __init__(self, base_dataset, reference_root, dataset_id, split, *,
                 expected_base=None, expected_screen=None):
        self.base_dataset = base_dataset
        self.dataset_id = str(dataset_id)
        self.split = str(split)
        self.reference_root = str(reference_root)
        self.base_meta, self.base = load_base(
            reference_root, dataset_id, split, expected=dict(expected_base or {}), mmap_mode='r'
        )
        self.screen_meta, self.screen = load_screen(
            reference_root, dataset_id, split, expected=dict(expected_screen or {}), mmap_mode='r'
        )
        if len(base_dataset) > int(self.base_meta['n']):
            raise ValueError(
                'DReSTG Reference shorter than runtime dataset: reference=%s dataset=%s' %
                (self.base_meta['n'], len(base_dataset))
            )

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        i = int(index)
        raw = screen_payload(self.screen, i)
        payload = {
            'pair': raw['pair'],
            'r': raw['r'],
            'sample_index': i,
            'statistics': self.base['stat'][i],
            'rho_full': self.base['rho'][i],
        }
        return (*self.base_dataset[i], payload)

    def inverse_transform(self, data):
        return self.base_dataset.inverse_transform(data)


def reference_collate(batch):
    base = [item[:4] for item in batch]
    x, y, xm, ym = default_collate(base)
    rels = [item[4] for item in batch]
    b = len(rels)
    c, l, two = rels[0]['statistics'].shape
    if two != 2:
        raise ValueError('DReSTG statistics last dimension must be 2')

    stats = torch.empty((b, c, l, 2), dtype=torch.float32)
    sample_index = torch.empty((b,), dtype=torch.long)
    full_u = c * (c-1)//2
    rho_full = torch.empty((b,full_u), dtype=torch.float32)
    total = sum(int(r['pair'].shape[0]) for r in rels)
    idx = torch.empty((total, 3), dtype=torch.long)
    rv = torch.empty((total,), dtype=torch.float32)
    offsets = torch.zeros((b + 1,), dtype=torch.long)

    pos = 0
    for bi, r in enumerate(rels):
        stats[bi] = torch.from_numpy(np.array(r['statistics'], dtype=np.float32, copy=True))
        sample_index[bi] = int(r['sample_index'])
        if np.asarray(r['rho_full']).shape != (full_u,):
            raise ValueError('Reference rho_full canonical upper-triangle shape mismatch')
        rho_full[bi] = torch.from_numpy(np.array(r['rho_full'], dtype=np.float32, copy=True))
        m = int(r['pair'].shape[0])
        if r['pair'].shape != (m, 2):
            raise ValueError('DReSTG packed pair shape invalid')
        if m:
            idx[pos:pos+m, 0] = bi
            idx[pos:pos+m, 1:] = torch.from_numpy(np.array(r['pair'], dtype=np.int64, copy=True))
            rv[pos:pos+m] = torch.from_numpy(np.array(r['r'], dtype=np.float32, copy=True))
            if torch.any(rv[pos:pos+m] <= 0) or torch.any(rv[pos:pos+m] > 1):
                raise ValueError('DReSTG packed pair_r must be selected raw positive Pearson in (0,1]')
        pos += m
        offsets[bi + 1] = pos

    return x, y, xm, ym, {
        'cache_kind': 'DRESTG-DUAL-RESOLUTION-V1',
        'packed': True,
        'sample_index': sample_index,
        'pair_index': idx,
        'pair_r': rv,
        'offsets': offsets,
        'statistics': stats,
        'rho_full': rho_full,
    }
