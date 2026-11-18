# ----------------------------------------------------------------------------
# Copyright (c) Yuan Feng et al., 2025; modified for DReSTG fine patching.
# License: MIT License
# ----------------------------------------------------------------------------
import torch


def pool_coarse_patch_nodes(fine_tokens, coarse_patch_count, fine_patches_per_coarse):
    """Pool all fine patches touching each coarse-window neighborhood.

    Each window owns m interior patches; the boundary fine patch is
    shared exactly once by both adjacent heads. No trainable gate.
    """
    l, m = int(coarse_patch_count), int(fine_patches_per_coarse)
    expected = l*m + l-1
    if fine_tokens.size(-2) != expected:
        raise ValueError('fine patch count inconsistent with coarse patches')
    coarse_nodes = []
    for i in range(l):
        base = i*(m+1)
        left = base - int(i > 0)
        right = base + m + int(i < l-1)
        coarse_nodes.append(fine_tokens[..., left:right, :].mean(dim=-2))
    return torch.stack(coarse_nodes, dim=-2)


def coarse_node_positions(coarse_patch_count, fine_patches_per_coarse):
    return [i*(int(fine_patches_per_coarse)+2) for i in range(int(coarse_patch_count))]


def fine_token_positions(coarse_patch_count, fine_patches_per_coarse):
    l, m = int(coarse_patch_count), int(fine_patches_per_coarse)
    coarse_nodes = set(coarse_node_positions(l, m))
    return [i for i in range(l*(m+2)-1) if i not in coarse_nodes]


def interleave_coarse_fine_tokens(coarse_nodes, fine_tokens, coarse_patch_count, fine_patches_per_coarse):
    """Interleave coarse patch nodes with a single time-ordered fine-token stream."""
    l, m = int(coarse_patch_count), int(fine_patches_per_coarse)
    if coarse_nodes.size(-2) != l or fine_tokens.size(-2) != l*m+l-1:
        raise ValueError('coarse patch nodes or fine token count mismatch')
    groups = []
    for i in range(l):
        start = i*(m+1)
        width = m + int(i < l-1)
        groups.extend((coarse_nodes[..., i:i+1, :], fine_tokens[..., start:start+width, :]))
    return torch.cat(groups, dim=-2)


def split_coarse_fine_tokens(total_tokens, coarse_patch_count, fine_patches_per_coarse):
    """Extract all fine tokens, including those straddling coarse-window seams."""
    h = coarse_node_positions(coarse_patch_count, fine_patches_per_coarse)
    f = fine_token_positions(coarse_patch_count, fine_patches_per_coarse)
    if total_tokens.size(-2) != len(h)+len(f):
        raise ValueError('total token count mismatch')
    return total_tokens[..., h, :], total_tokens[..., f, :]
