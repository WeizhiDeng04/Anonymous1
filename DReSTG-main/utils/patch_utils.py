# -----------------------------------------------------------------------------
# Copyright (c) Yuan Feng et al., 2025
# License: MIT License
# -----------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn.functional as F
import math

def compute_patch_layout(seq_len, coarse_patch_count, fine_patch_len, fine_patch_stride):
    k = math.ceil(((seq_len/coarse_patch_count)-fine_patch_len)/fine_patch_stride)
    if not k>=0:
        raise ValueError('patch_len too large')
    padding_len = int(coarse_patch_count * (fine_patch_len + (k * fine_patch_stride)) - seq_len)
    if padding_len >= int(fine_patch_len + (k * fine_patch_stride)):
        raise ValueError('stride too large')
    fine_patches_per_coarse = k+1
    interior_fine_patch_count = int(fine_patches_per_coarse * coarse_patch_count)
    
    return interior_fine_patch_count, fine_patches_per_coarse, padding_len

def extract_fine_patches(x_in, coarse_patch_count, fine_patch_len, fine_patch_stride):
    """Uniform fine patching over the entire normalized historical interval.

    Local-window seams have no tokenization privilege. With T=336, N_C=7 and
    P=2*S, the result has 55 (P12/S6) or 27 (P24/S12) fine tokens.
    """
    if x_in.size(-1) % coarse_patch_count:
        raise ValueError('sequence length must be divisible by local window count')
    if fine_patch_len != 2 * fine_patch_stride:
        raise ValueError('DReSTG requires 50% fine-patch overlap (P=2S)')
    window_length = x_in.size(-1) // coarse_patch_count
    if (window_length - fine_patch_len) % fine_patch_stride != 0:
        raise ValueError('fine patch grid must align with local window boundaries')
    return x_in.unfold(dimension=-1, size=fine_patch_len, step=fine_patch_stride)
