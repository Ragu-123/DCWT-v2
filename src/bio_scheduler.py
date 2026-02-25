"""
Biologically inspired scheduling utilities for DCWT-v2 training.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch.nn as nn

from .dcwt_v2 import DepthConditionedGWM, GatedWaveMerge


def _unwrap_module(module: nn.Module) -> nn.Module:
    return getattr(module, "_orig_mod", module)


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad = enabled


def set_gwm_depth_frozen(
    model,
    freeze_depths: Optional[Iterable[int]] = None,
    unfreeze_depths: Optional[Iterable[int]] = None,
) -> None:
    """
    Freeze/unfreeze GWM parameters by depth where possible.
    """
    freeze_set = set(freeze_depths or [])
    unfreeze_set = set(unfreeze_depths or [])

    for layer in model.layers:
        attn = layer.attention
        gwm = _unwrap_module(attn.gated_wave_merge)

        if isinstance(gwm, GatedWaveMerge):
            for depth in range(gwm.max_depth + 1):
                if depth in unfreeze_set:
                    enabled = True
                elif depth in freeze_set:
                    enabled = False
                else:
                    continue

                _set_requires_grad(gwm.gate_left[depth], enabled)
                _set_requires_grad(gwm.gate_right[depth], enabled)
                _set_requires_grad(gwm.layer_norms[depth], enabled)
                _set_requires_grad(gwm.skip_proj[depth], enabled)
                gwm.parent_query_init[depth].requires_grad = enabled
                gwm.skip_alpha[depth].requires_grad = enabled
            continue

        if isinstance(gwm, DepthConditionedGWM):
            # Shared GWM cannot be frozen per depth. Freeze all only when all depths are frozen.
            all_depths = set(range(attn.log_n + 1))
            enable_all = len(unfreeze_set) > 0
            disable_all = all_depths.issubset(freeze_set) and not enable_all
            if enable_all:
                _set_requires_grad(gwm, True)
            elif disable_all:
                _set_requires_grad(gwm, False)
            continue

        raise TypeError(f"Unsupported GWM module type: {type(gwm).__name__}")


def heartbeat_schedule(step: int, model) -> None:
    """
    Depth-staged unfreeze schedule.
    """
    max_depth = model.layers[0].attention.log_n
    depth_stop_1 = min(4, max_depth + 1)
    depth_stop_2 = min(7, max_depth + 1)

    if step == 0:
        set_gwm_depth_frozen(model, freeze_depths=range(max_depth + 1), unfreeze_depths=[])
    elif step == 500:
        set_gwm_depth_frozen(
            model,
            freeze_depths=range(depth_stop_1, max_depth + 1),
            unfreeze_depths=range(1, depth_stop_1),
        )
        print("Phase 1: unfreezing GWM depths 1-3")
    elif step == 2000:
        set_gwm_depth_frozen(
            model,
            freeze_depths=range(depth_stop_2, max_depth + 1),
            unfreeze_depths=range(depth_stop_1, depth_stop_2),
        )
        print("Phase 2: unfreezing GWM depths 4-6")
    elif step == 8000:
        set_gwm_depth_frozen(
            model,
            freeze_depths=[],
            unfreeze_depths=range(depth_stop_2, max_depth + 1),
        )
        print("Phase 3: all GWM depths active")


class SlimeMoldKScheduler:
    """
    Dynamic K allocation from accumulated gradient flow statistics.
    """

    def __init__(self, model, alpha: float = 0.3, mu: float = 0.3, update_every: int = 1000):
        self.model = model
        self.alpha = alpha
        self.mu = mu
        self.update_every = update_every
        self.grad_acc: Dict[int, float] = {}
        self.step_count = 0

        first_attn = model.layers[0].attention
        self.k_max = first_attn.k_max
        self.k_current: Dict[int, int] = {
            d: first_attn._k_at_depth(d) for d in range(first_attn.log_n + 1)
        }

    def accumulate(self, tree_grads: Dict[int, float]) -> None:
        for depth, grad_value in tree_grads.items():
            self.grad_acc[depth] = self.grad_acc.get(depth, 0.0) + float(abs(grad_value))
        self.step_count += 1

    def update(self) -> bool:
        if self.step_count < self.update_every or not self.grad_acc:
            return False

        max_flow = max(self.grad_acc.values()) + 1e-8
        new_k: Dict[int, int] = {}
        for depth, flow in self.grad_acc.items():
            ratio = flow / max_flow
            k_old = self.k_current.get(depth, self.k_max)
            k_new = k_old * (1.0 + self.alpha * (ratio - self.mu))
            new_k[depth] = max(1, min(self.k_max, int(round(k_new))))

        # Fill missing depths with previous schedule.
        first_attn = self.model.layers[0].attention
        for depth in range(first_attn.log_n + 1):
            new_k.setdefault(depth, self.k_current.get(depth, first_attn._k_at_depth(depth)))

        self.k_current = new_k
        self.grad_acc = {}
        self.step_count = 0

        for layer in self.model.layers:
            layer.attention.set_k_schedule(new_k)
        print(f"Slime mold K update: {new_k}")
        return True
