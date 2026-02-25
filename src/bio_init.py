"""
Biologically inspired initialization utilities for DCWT-v2.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from .dcwt_v2 import DepthConditionedGWM, GatedWaveMerge


def _unwrap_module(module: nn.Module) -> nn.Module:
    # torch.compile wraps modules in an OptimizedModule that exposes _orig_mod.
    return getattr(module, "_orig_mod", module)


def _make_haar_queries(k: int, head_dim: int) -> torch.Tensor:
    if k <= 0:
        return torch.zeros(0, head_dim)

    basis = torch.zeros(k, head_dim, dtype=torch.float32)
    basis[0, :] = 1.0 / math.sqrt(max(head_dim, 1))
    if k > 1:
        half = max(head_dim // 2, 1)
        basis[1, :half] = 1.0 / math.sqrt(half)
        basis[1, half:] = -1.0 / math.sqrt(max(head_dim - half, 1))

    for i in range(2, k):
        v = torch.randn(head_dim, dtype=torch.float32)
        for j in range(i):
            v = v - torch.dot(v, basis[j]) * basis[j]
        norm = v.norm().clamp_min(1e-8)
        basis[i] = v / norm
    return basis


def init_gwm_haar(gwm_module: nn.Module, depth_from_leaf: int, k_parent: int, head_dim: int) -> None:
    """
    Initialize one GWM depth to a Haar-like merge prior.
    """
    gwm = _unwrap_module(gwm_module)

    with torch.no_grad():
        if isinstance(gwm, GatedWaveMerge):
            d = depth_from_leaf
            gwm.gate_left[d].weight.zero_()
            gwm.gate_left[d].bias.fill_(4.6)
            gwm.gate_right[d].weight.zero_()
            gwm.gate_right[d].bias.fill_(4.6)

            haar = _make_haar_queries(k_parent, head_dim).to(gwm.parent_query_init[d].dtype)
            gwm.parent_query_init[d][:k_parent].copy_(haar)
            gwm.skip_alpha[d].fill_(-3.0)
            return

        if isinstance(gwm, DepthConditionedGWM):
            gwm.gate_left.weight.zero_()
            gwm.gate_left.bias.fill_(4.6)
            gwm.gate_right.weight.zero_()
            gwm.gate_right.bias.fill_(4.6)
            gwm.query_offset.weight.zero_()
            gwm.query_offset.bias.zero_()

            haar = _make_haar_queries(gwm.k_max, head_dim).to(gwm.parent_query_base.dtype)
            gwm.parent_query_base.copy_(haar)
            gwm.skip_alpha.fill_(-3.0)
            return

        raise TypeError(f"Unsupported GWM module type: {type(gwm).__name__}")


def init_all_depths_haar(model: Any) -> None:
    """
    Apply Haar-like initialization across all DCWT-v2 layers.
    """
    for layer in model.layers:
        attn = layer.attention
        gwm = _unwrap_module(attn.gated_wave_merge)
        if isinstance(gwm, GatedWaveMerge):
            for depth in range(1, attn.log_n + 1):
                k_parent = min(2 ** depth, attn.k_max)
                init_gwm_haar(attn.gated_wave_merge, depth, k_parent, attn.head_dim)
        elif isinstance(gwm, DepthConditionedGWM):
            init_gwm_haar(attn.gated_wave_merge, 1, gwm.k_max, attn.head_dim)
        else:
            raise TypeError(f"Unsupported GWM module type: {type(gwm).__name__}")
