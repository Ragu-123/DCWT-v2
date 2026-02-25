"""
DCWT-v2 core implementation.

This module implements the architecture proposed in newproposal.txt:
- Gated Wave Merge (GWM)
- Multi-Vector Node Bank (MVNB)
- Complementary Dual Coverage (CDC: local window + tree)
- Depth-Decomposed Queries (DDQ)
- Tree LayerNorm + skip connections
"""

from __future__ import annotations

import functools
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton  # noqa: F401
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


def k_at_depth(d_leaf: int, k_max: int) -> int:
    """Number of vectors stored at a node for a given leaf-depth."""
    return min(2 ** max(d_leaf, 0), k_max)


class CausalSegmentTree:
    """1-indexed segment tree helpers for causal prefix cover sets."""

    def __init__(self, max_len: int):
        self.max_len = max_len
        self.log_n = math.ceil(math.log2(max_len + 1))
        self.tree_size = 2 ** (self.log_n + 1)
        self.leaf_start = self.tree_size // 2

    def leaf_index(self, pos: int) -> int:
        return self.leaf_start + pos

    def depth_from_leaf(self, node_idx: int) -> int:
        # Leaf depth = 0, root depth = log_n.
        d_from_root = int(math.floor(math.log2(max(node_idx, 1))))
        return self.log_n - d_from_root

    def cover_set_with_depth(self, pos: int) -> List[Tuple[int, int]]:
        """
        O(log n) node indices covering prefix [0..pos-1], each with leaf-depth.
        """
        if pos <= 0:
            return []

        l_idx = self.leaf_start
        r_idx = self.leaf_start + min(pos, self.max_len)
        out: List[Tuple[int, int]] = []

        while l_idx < r_idx:
            if l_idx & 1:
                out.append((l_idx, self.depth_from_leaf(l_idx)))
                l_idx += 1
            if r_idx & 1:
                r_idx -= 1
                out.append((r_idx, self.depth_from_leaf(r_idx)))
            l_idx >>= 1
            r_idx >>= 1

        return out


@functools.lru_cache(maxsize=16)
def _get_tree_structure(
    max_len: int, device_str: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Cached internal tree connectivity for Jacobi updates.

    Returns:
    - node_idx: internal node indices
    - left_idx: left child index per internal node
    - right_idx: right child index per internal node
    - depth_idx: depth-from-leaf per internal node
    """
    cst = CausalSegmentTree(max_len)
    leaf_start = cst.leaf_start
    internals = list(range(1, leaf_start))
    lefts = [2 * n for n in internals]
    rights = [2 * n + 1 for n in internals]
    depths = [cst.depth_from_leaf(n) for n in internals]

    device = torch.device(device_str)
    return (
        torch.tensor(internals, dtype=torch.long, device=device),
        torch.tensor(lefts, dtype=torch.long, device=device),
        torch.tensor(rights, dtype=torch.long, device=device),
        torch.tensor(depths, dtype=torch.long, device=device),
    )


class GatedWaveMerge(nn.Module):
    """
    Gated merge that builds multi-vector parent nodes.

    Input:
      - f_left:  (B, H, K_left, D)
      - f_right: (B, H, K_right, D)
    Output:
      - f_parent: (B, H, K_parent, D), K_parent <= k_max
    """

    def __init__(self, num_heads: int, head_dim: int, max_depth: int, k_max: int):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("DCWT-v2 requires even head_dim for complex wave rotation.")

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_depth = max_depth
        self.k_max = k_max

        self.wave_freq_base = nn.Parameter(torch.linspace(0.3, 4.0, num_heads))
        self.wave_damp_base = nn.Parameter(torch.linspace(-3.0, 0.5, num_heads))
        self.wave_phase_base = nn.Parameter(torch.linspace(0.0, math.pi, num_heads))

        self.gate_left = nn.ModuleList(
            [nn.Linear(head_dim * 2, head_dim, bias=True) for _ in range(max_depth + 1)]
        )
        self.gate_right = nn.ModuleList(
            [nn.Linear(head_dim * 2, head_dim, bias=True) for _ in range(max_depth + 1)]
        )

        self.parent_query_init = nn.ParameterList(
            [nn.Parameter(torch.randn(k_max, head_dim) * 0.02) for _ in range(max_depth + 1)]
        )

        self.layer_norms = nn.ModuleList([nn.LayerNorm(head_dim) for _ in range(max_depth + 1)])
        self.skip_alpha = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.1, dtype=torch.float32)) for _ in range(max_depth + 1)]
        )
        self.skip_proj = nn.ModuleList(
            [nn.Linear(head_dim, head_dim, bias=False) for _ in range(max_depth + 1)]
        )

        for depth in range(max_depth + 1):
            nn.init.zeros_(self.gate_left[depth].weight)
            nn.init.constant_(self.gate_left[depth].bias, 0.5)
            nn.init.zeros_(self.gate_right[depth].weight)
            nn.init.constant_(self.gate_right[depth].bias, 0.5)

    def _rotate_right(self, f: torch.Tensor, depth: int) -> torch.Tensor:
        """
        Apply depth-scaled complex rotation to right-branch vectors.
        Uses Triton kernel when available on CUDA.
        """
        alpha_base = F.softplus(self.wave_damp_base)  # (H,)
        omega_base = self.wave_freq_base
        phi_base = self.wave_phase_base

        scale = max(2 ** (self.max_depth - depth), 1)
        delta_t = float(scale)
        alpha_d = alpha_base / scale
        omega_d = omega_base / scale
        phi_d = phi_base + depth * (math.pi / 4.0)

        decay = torch.exp(-alpha_d * delta_t)
        phi_real = decay * torch.cos(omega_d * delta_t + phi_d)
        phi_imag = decay * torch.sin(omega_d * delta_t + phi_d)

        if TRITON_AVAILABLE and f.is_cuda:
            try:
                from .triton_wave_rotate import wave_rotate_triton

                return wave_rotate_triton(f, phi_real, phi_imag)
            except Exception:
                # Fall through to the PyTorch implementation.
                pass

        d2 = self.head_dim // 2
        f_re = f[..., :d2]
        f_im = f[..., d2:]

        if f.dim() == 4:
            pr = phi_real.view(1, self.num_heads, 1, 1)
            pi_ = phi_imag.view(1, self.num_heads, 1, 1)
        elif f.dim() == 5:
            pr = phi_real.view(1, 1, self.num_heads, 1, 1)
            pi_ = phi_imag.view(1, 1, self.num_heads, 1, 1)
        else:
            raise ValueError(f"Unexpected tensor rank for rotation: {f.dim()}")

        rot_re = pr * f_re - pi_ * f_im
        rot_im = pi_ * f_re + pr * f_im
        return torch.cat([rot_re, rot_im], dim=-1)

    def forward(self, f_left: torch.Tensor, f_right: torch.Tensor, depth: int) -> torch.Tensor:
        bsz, n_heads, k_left, dim = f_left.shape
        _, _, k_right, _ = f_right.shape
        if n_heads != self.num_heads or dim != self.head_dim:
            raise ValueError("Unexpected tensor shape in GatedWaveMerge.")

        rotated_right = self._rotate_right(f_right, depth)

        # Content-adaptive gates from pooled child summaries.
        left_mean = f_left.mean(dim=2)
        right_mean = rotated_right.mean(dim=2)
        gate_in = torch.cat([left_mean, right_mean], dim=-1).reshape(bsz * n_heads, 2 * dim)

        g_l = torch.sigmoid(self.gate_left[depth](gate_in)).view(bsz, n_heads, 1, dim)
        g_r = torch.sigmoid(self.gate_right[depth](gate_in)).view(bsz, n_heads, 1, dim)

        bank = torch.cat([f_left * g_l, rotated_right * g_r], dim=2)
        k_parent = min(k_left + k_right, self.k_max)

        parent_q = self.parent_query_init[depth][:k_parent]
        parent_q = parent_q.unsqueeze(0).unsqueeze(0).expand(bsz, n_heads, -1, -1)

        attn = torch.einsum("bhqd,bhkd->bhqk", parent_q, bank) / math.sqrt(dim)
        attn = F.softmax(attn, dim=-1)
        parent_raw = torch.einsum("bhqk,bhkd->bhqd", attn, bank)

        parent_norm = self.layer_norms[depth](parent_raw.reshape(bsz * n_heads * k_parent, dim))
        parent_norm = parent_norm.view(bsz, n_heads, k_parent, dim)

        skip = self.skip_proj[depth](left_mean.reshape(bsz * n_heads, dim)).view(bsz, n_heads, 1, dim)
        alpha = torch.sigmoid(self.skip_alpha[depth])
        out = parent_norm + alpha * skip
        if out.dtype != f_left.dtype:
            out = out.to(f_left.dtype)
        return out

    def forward_batched(
        self,
        f_left: torch.Tensor,   # (B, M, H, K_left, D)
        f_right: torch.Tensor,  # (B, M, H, K_right, D)
        depth: int,
    ) -> torch.Tensor:
        """
        Batched merge for all nodes at a tree level.
        Mathematically equivalent to calling forward() M times.
        """
        bsz, n_nodes, n_heads, k_left, dim = f_left.shape
        _, _, _, k_right, _ = f_right.shape
        if n_heads != self.num_heads or dim != self.head_dim:
            raise ValueError("Unexpected tensor shape in GatedWaveMerge.forward_batched.")

        left_flat = f_left.reshape(bsz * n_nodes, n_heads, k_left, dim)
        right_flat = f_right.reshape(bsz * n_nodes, n_heads, k_right, dim)
        merged_flat = self.forward(left_flat, right_flat, depth)

        k_parent = merged_flat.shape[2]
        return merged_flat.reshape(bsz, n_nodes, n_heads, k_parent, dim)


class DepthConditionedGWM(nn.Module):
    """
    Shared GWM conditioned on depth embedding.

    Input:
      - f_left:  (B, H, K_left, D)
      - f_right: (B, H, K_right, D)
      - depth:   depth-from-leaf
    Output:
      - f_parent: (B, H, K_parent, D)
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        k_max: int,
        max_depth: int,
        depth_embed_dim: int = 32,
    ):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("DCWT-v2 requires even head_dim for complex wave rotation.")

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.k_max = k_max
        self.max_depth = max_depth
        self.depth_embed_dim = depth_embed_dim

        input_dim = 2 * head_dim + depth_embed_dim
        self.gate_left = nn.Linear(input_dim, head_dim, bias=True)
        self.gate_right = nn.Linear(input_dim, head_dim, bias=True)
        self.parent_query_base = nn.Parameter(torch.randn(k_max, head_dim) * 0.02)
        self.query_offset = nn.Linear(depth_embed_dim, k_max * head_dim, bias=True)
        self.layer_norm = nn.LayerNorm(head_dim)
        self.skip_proj = nn.Linear(head_dim, head_dim, bias=False)
        self.skip_alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        self.wave_freq_base = nn.Parameter(torch.linspace(0.3, 4.0, num_heads))
        self.wave_damp_base = nn.Parameter(torch.linspace(-3.0, 0.5, num_heads))
        self.wave_phase_base = nn.Parameter(torch.linspace(0.0, math.pi, num_heads))

        nn.init.zeros_(self.gate_left.weight)
        nn.init.constant_(self.gate_left.bias, 0.5)
        nn.init.zeros_(self.gate_right.weight)
        nn.init.constant_(self.gate_right.bias, 0.5)
        nn.init.zeros_(self.query_offset.weight)
        nn.init.zeros_(self.query_offset.bias)

        depth_embed = self._make_sin_embed(max_depth + 1, depth_embed_dim)
        self.register_buffer("depth_embed", depth_embed, persistent=False)

    @staticmethod
    def _make_sin_embed(n: int, d_model: int) -> torch.Tensor:
        pos = torch.arange(n, dtype=torch.float32).unsqueeze(1)
        inv = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / max(d_model, 1))
        )
        embed = torch.zeros(n, d_model, dtype=torch.float32)
        embed[:, 0::2] = torch.sin(pos * inv)
        embed[:, 1::2] = torch.cos(pos * inv)
        return embed

    def _rotate_right(self, f: torch.Tensor, depth: int) -> torch.Tensor:
        alpha_base = F.softplus(self.wave_damp_base)  # (H,)
        omega_base = self.wave_freq_base
        phi_base = self.wave_phase_base

        scale = max(2 ** (self.max_depth - depth), 1)
        delta_t = float(scale)
        alpha_d = alpha_base / scale
        omega_d = omega_base / scale
        phi_d = phi_base + depth * (math.pi / 4.0)

        decay = torch.exp(-alpha_d * delta_t)
        phi_real = decay * torch.cos(omega_d * delta_t + phi_d)
        phi_imag = decay * torch.sin(omega_d * delta_t + phi_d)

        if TRITON_AVAILABLE and f.is_cuda:
            try:
                from .triton_wave_rotate import wave_rotate_triton

                return wave_rotate_triton(f, phi_real, phi_imag)
            except Exception:
                pass

        d2 = self.head_dim // 2
        f_re = f[..., :d2]
        f_im = f[..., d2:]
        if f.dim() != 4:
            raise ValueError(f"Unexpected tensor rank for rotation: {f.dim()}")

        pr = phi_real.view(1, self.num_heads, 1, 1)
        pi_ = phi_imag.view(1, self.num_heads, 1, 1)
        rot_re = pr * f_re - pi_ * f_im
        rot_im = pi_ * f_re + pr * f_im
        return torch.cat([rot_re, rot_im], dim=-1)

    def forward(self, f_left: torch.Tensor, f_right: torch.Tensor, depth: int) -> torch.Tensor:
        bsz, n_heads, k_left, dim = f_left.shape
        _, _, k_right, _ = f_right.shape
        if n_heads != self.num_heads or dim != self.head_dim:
            raise ValueError("Unexpected tensor shape in DepthConditionedGWM.")

        rotated_right = self._rotate_right(f_right, depth)
        k_parent = min(k_left + k_right, self.k_max)

        left_mean = f_left.mean(dim=2)
        right_mean = rotated_right.mean(dim=2)
        depth_vec = self.depth_embed[depth].to(left_mean.dtype)
        depth_vec = depth_vec.view(1, 1, -1).expand(bsz, n_heads, -1)

        gate_in = torch.cat([left_mean, right_mean, depth_vec], dim=-1)
        gate_in_flat = gate_in.reshape(bsz * n_heads, -1)
        g_l = torch.sigmoid(self.gate_left(gate_in_flat)).view(bsz, n_heads, 1, dim)
        g_r = torch.sigmoid(self.gate_right(gate_in_flat)).view(bsz, n_heads, 1, dim)

        bank = torch.cat([f_left * g_l, rotated_right * g_r], dim=2)

        q_offset = self.query_offset(depth_vec[0, 0]).view(self.k_max, dim).to(bank.dtype)
        parent_q = self.parent_query_base[:k_parent].to(bank.dtype) + q_offset[:k_parent]
        parent_q = parent_q.unsqueeze(0).unsqueeze(0).expand(bsz, n_heads, -1, -1)

        attn = torch.einsum("bhqd,bhkd->bhqk", parent_q, bank) / math.sqrt(dim)
        attn = F.softmax(attn, dim=-1)
        parent_raw = torch.einsum("bhqk,bhkd->bhqd", attn, bank)

        parent_norm = self.layer_norm(parent_raw.reshape(bsz * n_heads * k_parent, dim))
        parent_norm = parent_norm.view(bsz, n_heads, k_parent, dim)

        skip = self.skip_proj(left_mean.reshape(bsz * n_heads, dim)).view(bsz, n_heads, 1, dim)
        alpha = torch.sigmoid(self.skip_alpha)
        out = parent_norm + alpha * skip
        if out.dtype != f_left.dtype:
            out = out.to(f_left.dtype)
        return out


class DepthDecomposedQuery(nn.Module):
    """Depth-specific query projections and learnable depth temperatures."""

    def __init__(self, num_heads: int, head_dim: int, max_depth: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.depth_proj = nn.ModuleList(
            [nn.Linear(head_dim, head_dim, bias=False) for _ in range(max_depth + 1)]
        )
        self.depth_temp = nn.Parameter(torch.linspace(0.5, 4.0, max_depth + 1))

    def get_query(self, q_base: torch.Tensor, depth: int) -> torch.Tensor:
        bsz, n_heads, dim = q_base.shape
        q_delta = self.depth_proj[depth](q_base.reshape(bsz * n_heads, dim)).view(bsz, n_heads, dim)
        return q_base + q_delta

    def get_scale(self, depth: int) -> torch.Tensor:
        temp = F.softplus(self.depth_temp[depth]) + 1e-6
        return 1.0 / (temp * math.sqrt(self.head_dim))


class DCWTv2Attention(nn.Module):
    """
    DCWT-v2 attention:
    local exact attention + multi-vector causal tree attention.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        max_seq_len: int,
        k_max: int = 8,
        local_window: int = 32,
        tree_mode: str = "flash_only",
        dropout: float = 0.1,
        use_depth_conditioned_gwm: bool = False,
        depth_embed_dim: int = 32,
        compile_gwm: bool = True,
    ):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.max_seq_len = max_seq_len
        self.k_max = k_max
        self.local_window = local_window
        self.use_depth_conditioned_gwm = use_depth_conditioned_gwm
        if tree_mode not in {"full", "local_only", "fast_hybrid", "flash_only"}:
            raise ValueError(
                "tree_mode must be one of: {'full', 'local_only', 'fast_hybrid', 'flash_only'}"
            )
        self.tree_mode = tree_mode

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        self.k_local_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_local_proj = nn.Linear(embedding_dim, embedding_dim)
        # Fast-hybrid global branch: kernelized causal linear attention.
        self.global_feature_dim = min(8, self.head_dim)
        self.q_global_proj = nn.Linear(
            embedding_dim, num_heads * self.global_feature_dim
        )
        self.k_global_proj = nn.Linear(
            embedding_dim, num_heads * self.global_feature_dim
        )
        self.v_global_proj = nn.Linear(embedding_dim, embedding_dim)
        self.hybrid_alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        self.cst = CausalSegmentTree(max_seq_len)
        self.log_n = self.cst.log_n
        self.leaf_start = self.cst.leaf_start

        if self.use_depth_conditioned_gwm:
            self.gated_wave_merge = DepthConditionedGWM(
                num_heads=num_heads,
                head_dim=self.head_dim,
                k_max=k_max,
                max_depth=self.log_n,
                depth_embed_dim=depth_embed_dim,
            )
        else:
            self.gated_wave_merge = GatedWaveMerge(
                num_heads=num_heads,
                head_dim=self.head_dim,
                max_depth=self.log_n,
                k_max=k_max,
            )
        if compile_gwm and hasattr(torch, "compile") and torch.cuda.is_available():
            try:
                self.gated_wave_merge = torch.compile(
                    self.gated_wave_merge,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
            except Exception:
                pass
        self.ddq = DepthDecomposedQuery(num_heads, self.head_dim, self.log_n)

        self.cross_head_coupling = nn.ParameterList(
            [nn.Parameter(torch.eye(num_heads) + 0.01 * torch.randn(num_heads, num_heads)) for _ in range(self.log_n + 1)]
        )

        self.gate_proj = nn.Linear(embedding_dim, embedding_dim)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 2.0)

        self.dropout = nn.Dropout(dropout)
        self._local_mask_cache: Dict[Tuple[int, int, str], torch.Tensor] = {}
        self._cover_table_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._cover_table_overflow_cache: Dict[
            Tuple[int, str], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        self._cover_table_master = self._build_cover_table(self.max_seq_len, torch.device("cpu"))
        self.k_current: Dict[int, int] = {
            d: k_at_depth(d, self.k_max) for d in range(self.log_n + 1)
        }

    def _apply_cross_head_coupling(self, f: torch.Tensor, depth: int) -> torch.Tensor:
        coupling = F.softmax(self.cross_head_coupling[depth], dim=-1).to(f.dtype)  # (H, H)
        return torch.einsum("ij,bjkd->bikd", coupling, f).to(f.dtype)

    def _k_at_depth(self, depth: int) -> int:
        return int(self.k_current.get(depth, k_at_depth(depth, self.k_max)))

    def set_k_schedule(self, k_schedule: Dict[int, int]) -> None:
        updated: Dict[int, int] = {}
        for d in range(self.log_n + 1):
            target = int(k_schedule.get(d, self._k_at_depth(d)))
            updated[d] = max(1, min(self.k_max, target))
        self.k_current = updated

    def get_tree_grad_norms(self) -> Dict[int, float]:
        out: Dict[int, float] = {}
        for d in range(self.log_n + 1):
            norms: List[float] = []

            coupling_grad = self.cross_head_coupling[d].grad
            if coupling_grad is not None:
                norms.append(float(coupling_grad.norm().item()))

            ddq_grad = self.ddq.depth_proj[d].weight.grad
            if ddq_grad is not None:
                norms.append(float(ddq_grad.norm().item()))

            if isinstance(self.gated_wave_merge, GatedWaveMerge):
                for tensor in [
                    self.gated_wave_merge.gate_left[d].weight.grad,
                    self.gated_wave_merge.gate_left[d].bias.grad,
                    self.gated_wave_merge.gate_right[d].weight.grad,
                    self.gated_wave_merge.gate_right[d].bias.grad,
                    self.gated_wave_merge.parent_query_init[d].grad,
                    self.gated_wave_merge.skip_alpha[d].grad,
                    self.gated_wave_merge.skip_proj[d].weight.grad,
                ]:
                    if tensor is not None:
                        norms.append(float(tensor.norm().item()))
            else:
                for tensor in [
                    self.gated_wave_merge.gate_left.weight.grad,
                    self.gated_wave_merge.gate_left.bias.grad,
                    self.gated_wave_merge.gate_right.weight.grad,
                    self.gated_wave_merge.gate_right.bias.grad,
                    self.gated_wave_merge.parent_query_base.grad,
                    self.gated_wave_merge.query_offset.weight.grad,
                    self.gated_wave_merge.query_offset.bias.grad,
                    self.gated_wave_merge.skip_alpha.grad,
                    self.gated_wave_merge.skip_proj.weight.grad,
                ]:
                    if tensor is not None:
                        norms.append(float(tensor.norm().item()))

            if norms:
                out[d] = sum(norms) / len(norms)
        return out

    def _build_tree_training(self, v: torch.Tensor) -> torch.Tensor:
        """
        Build a dense tree tensor using level-parallel merges.

        v: (B, N, H, D_head)
        returns: (B, T, H, K_max, D_head), T = self.tree_size
        """
        bsz, seq_len, n_heads, dim = v.shape
        tree_size = self.cst.tree_size

        # Dense tree storage.
        tree = v.new_zeros(bsz, tree_size, n_heads, self.k_max, dim)
        tree_dtype = tree.dtype
        node_present = torch.zeros(tree_size, dtype=torch.bool, device=v.device)

        # Insert leaves (K=1).
        leaf_start = self.leaf_start
        leaf_end = leaf_start + seq_len
        tree[:, leaf_start:leaf_end, :, :1, :] = v.unsqueeze(3)
        node_present[leaf_start:leaf_end] = True

        # Level-parallel bottom-up merge: one batched merge call per depth.
        for d_from_leaf in range(1, self.log_n + 1):
            d_from_root = self.log_n - d_from_leaf
            node_start = 2 ** d_from_root
            node_end = 2 ** (d_from_root + 1)
            n_nodes = node_end - node_start
            if n_nodes <= 0:
                continue

            left_start = 2 * node_start
            right_start = left_start + 1

            left_idx = torch.arange(left_start, left_start + 2 * n_nodes, 2, device=v.device)
            right_idx = left_idx + 1
            has_left = node_present[left_idx]
            has_right = node_present[right_idx]
            has_any = has_left | has_right
            if not bool(has_any.any()):
                continue

            k_child = self._k_at_depth(d_from_leaf - 1)
            f_left = tree[:, left_start:left_start + 2 * n_nodes:2, :, :k_child, :]
            f_right = tree[:, right_start:right_start + 2 * n_nodes:2, :, :k_child, :]

            left_flat = f_left.reshape(bsz * n_nodes, n_heads, k_child, dim)
            right_flat = f_right.reshape(bsz * n_nodes, n_heads, k_child, dim)
            merged = self.gated_wave_merge(left_flat, right_flat, d_from_leaf)
            merged = merged.reshape(bsz, n_nodes, n_heads, -1, dim).to(tree_dtype)
            k_parent = min(merged.shape[3], self._k_at_depth(d_from_leaf))

            parent = v.new_zeros(bsz, n_nodes, n_heads, self.k_max, dim)
            both = has_left & has_right
            left_only = has_left & ~has_right
            right_only = has_right & ~has_left

            if bool(both.any()):
                parent[:, both, :, :k_parent, :] = merged[:, both, :, :k_parent, :]
            if bool(left_only.any()):
                parent[:, left_only, :, :k_child, :] = f_left[:, left_only, :, :, :]
            if bool(right_only.any()):
                parent[:, right_only, :, :k_child, :] = f_right[:, right_only, :, :, :]

            coupling = F.softmax(self.cross_head_coupling[d_from_leaf], dim=-1).to(tree_dtype)
            parent = torch.einsum("ij,bmjkd->bmikd", coupling, parent).to(tree_dtype)

            tree[:, node_start:node_end, :, :, :] = parent
            node_present[node_start:node_end] = has_any

        return tree

    def _build_tree_jacobi(self, v: torch.Tensor, jacobi_iters: int = 2) -> torch.Tensor:
        """
        Parallel Jacobi tree update using previous-iteration child values.

        v: (B, N, H, D_head)
        returns: (B, T, H, K_max, D_head)
        """
        if jacobi_iters <= 0:
            return self._build_tree_training(v)

        bsz, seq_len, n_heads, dim = v.shape
        tree_size = self.cst.tree_size
        tree = v.new_zeros(bsz, tree_size, n_heads, self.k_max, dim)
        tree_dtype = tree.dtype

        node_present = torch.zeros(tree_size, dtype=torch.bool, device=v.device)
        leaf_start = self.leaf_start
        leaf_end = leaf_start + seq_len
        tree[:, leaf_start:leaf_end, :, :1, :] = v.unsqueeze(3)
        node_present[leaf_start:leaf_end] = True

        # Haar-like bootstrap on internals to start Jacobi close to useful structure.
        with torch.no_grad():
            for d_from_leaf in range(1, self.log_n + 1):
                d_from_root = self.log_n - d_from_leaf
                node_start = 2 ** d_from_root
                node_end = 2 ** (d_from_root + 1)
                n_nodes = node_end - node_start
                if n_nodes <= 0:
                    continue

                left_idx = torch.arange(2 * node_start, 2 * node_end, 2, device=v.device)
                right_idx = left_idx + 1
                has_left = node_present[left_idx]
                has_right = node_present[right_idx]
                has_any = has_left | has_right
                if not bool(has_any.any()):
                    continue

                k_child = self._k_at_depth(d_from_leaf - 1)
                k_parent = self._k_at_depth(d_from_leaf)
                f_left = tree[:, left_idx, :, :k_child, :]
                f_right = tree[:, right_idx, :, :k_child, :]
                parent = v.new_zeros(bsz, n_nodes, n_heads, self.k_max, dim)

                both = has_left & has_right
                left_only = has_left & ~has_right
                right_only = has_right & ~has_left
                if bool(both.any()):
                    k_boot = min(k_parent, k_child)
                    parent[:, both, :, :k_boot, :] = (
                        f_left[:, both, :, :k_boot, :] + f_right[:, both, :, :k_boot, :]
                    ) / math.sqrt(2.0)
                if bool(left_only.any()):
                    parent[:, left_only, :, :k_child, :] = f_left[:, left_only, :, :, :]
                if bool(right_only.any()):
                    parent[:, right_only, :, :k_child, :] = f_right[:, right_only, :, :, :]

                tree[:, node_start:node_end, :, :, :] = parent
                node_present[node_start:node_end] = has_any

        node_idx, left_idx, right_idx, depth_idx = _get_tree_structure(
            self.max_seq_len, str(v.device)
        )

        for _ in range(jacobi_iters):
            prev_tree = tree
            prev_present = node_present
            tree = prev_tree.clone()
            node_present = prev_present.clone()

            for depth in range(1, self.log_n + 1):
                depth_mask = depth_idx == depth
                if not bool(depth_mask.any()):
                    continue

                idx_d = node_idx[depth_mask]
                left_d = left_idx[depth_mask]
                right_d = right_idx[depth_mask]

                has_left = prev_present[left_d]
                has_right = prev_present[right_d]
                has_any = has_left | has_right
                if not bool(has_any.any()):
                    continue

                k_child = self._k_at_depth(depth - 1)
                f_left = prev_tree[:, left_d, :, :k_child, :]
                f_right = prev_tree[:, right_d, :, :k_child, :]
                n_nodes = idx_d.numel()
                parent = v.new_zeros(bsz, n_nodes, n_heads, self.k_max, dim)

                both = has_left & has_right
                left_only = has_left & ~has_right
                right_only = has_right & ~has_left

                if bool(both.any()):
                    left_b = f_left[:, both, :, :, :]
                    right_b = f_right[:, both, :, :, :]
                    n_both = left_b.shape[1]
                    left_flat = left_b.reshape(bsz * n_both, n_heads, k_child, dim)
                    right_flat = right_b.reshape(bsz * n_both, n_heads, k_child, dim)
                    merged = self.gated_wave_merge(left_flat, right_flat, depth)
                    merged = merged.reshape(bsz, n_both, n_heads, -1, dim).to(tree_dtype)
                    k_parent = min(merged.shape[3], self._k_at_depth(depth))
                    parent[:, both, :, :k_parent, :] = merged[:, :, :, :k_parent, :]

                if bool(left_only.any()):
                    parent[:, left_only, :, :k_child, :] = f_left[:, left_only, :, :, :]
                if bool(right_only.any()):
                    parent[:, right_only, :, :k_child, :] = f_right[:, right_only, :, :, :]

                coupling = F.softmax(self.cross_head_coupling[depth], dim=-1).to(tree_dtype)
                parent = torch.einsum("ij,bmjkd->bmikd", coupling, parent).to(tree_dtype)

                tree[:, idx_d, :, :, :] = parent
                node_present[idx_d] = has_any

        return tree

    def _get_local_mask(self, n_tokens: int, window: int, device: torch.device) -> torch.Tensor:
        key = (n_tokens, window, str(device))
        cached = self._local_mask_cache.get(key)
        if cached is not None:
            return cached

        rows = torch.arange(n_tokens, device=device).unsqueeze(1)
        cols = torch.arange(n_tokens, device=device).unsqueeze(0)
        mask = (cols < rows) & (cols >= rows - window)
        self._local_mask_cache[key] = mask
        return mask

    def _local_attention(self, q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, n_heads, dim = q.shape
        window = min(self.local_window, seq_len)

        k_local = self.k_local_proj(x).view(bsz, seq_len, n_heads, dim)
        v_local = self.v_local_proj(x).view(bsz, seq_len, n_heads, dim)

        # SDPA expects (B, H, N, D).
        q_t = q.transpose(1, 2)
        k_t = k_local.transpose(1, 2)
        v_t = v_local.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            attn_mask=self._get_local_mask(seq_len, window, q.device),
            dropout_p=0.0,
            scale=1.0 / math.sqrt(dim),
        )
        out = torch.nan_to_num(out, nan=0.0)
        return out.transpose(1, 2)

    def _flash_causal_attention(self, q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Fast path that lets SDPA pick fused causal kernels (Flash/MemEfficient).
        """
        bsz, seq_len, n_heads, dim = q.shape
        k_local = self.k_local_proj(x).view(bsz, seq_len, n_heads, dim)
        v_local = self.v_local_proj(x).view(bsz, seq_len, n_heads, dim)

        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k_local.transpose(1, 2),
            v_local.transpose(1, 2),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=1.0 / math.sqrt(dim),
        )
        out = torch.nan_to_num(out, nan=0.0)
        return out.transpose(1, 2)

    def _linear_global_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Causal linear attention with prefix-sum KV states.
        Complexity: O(B * N * H * R * D), where R is a small feature rank.
        """
        bsz, seq_len, _ = x.shape
        n_heads = self.num_heads
        dim = self.head_dim
        rank = self.global_feature_dim

        q_feat = self.q_global_proj(x).view(bsz, seq_len, n_heads, rank)
        k_feat = self.k_global_proj(x).view(bsz, seq_len, n_heads, rank)
        v = self.v_global_proj(x).view(bsz, seq_len, n_heads, dim)

        # Keep prefix sums in fp32 for stability under autocast.
        work_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        q_phi = (F.elu(q_feat.to(work_dtype)) + 1.0).clamp_min(1e-4)
        k_phi = (F.elu(k_feat.to(work_dtype)) + 1.0).clamp_min(1e-4)
        v_work = v.to(work_dtype)

        kv = torch.einsum("bnhr,bnhd->bnhrd", k_phi, v_work)
        kv_prefix = torch.cumsum(kv, dim=1)
        k_prefix = torch.cumsum(k_phi, dim=1)

        numerator = torch.einsum("bnhr,bnhrd->bnhd", q_phi, kv_prefix)
        denominator = (q_phi * k_prefix).sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return (numerator / denominator).to(x.dtype)

    def _build_cover_table(
        self, n_tokens: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_cover = self.log_n + 1
        node_indices = torch.full(
            (n_tokens, max_cover), -1, dtype=torch.long, device=device
        )
        node_depths = torch.zeros((n_tokens, max_cover), dtype=torch.long, device=device)
        valid_mask = torch.zeros((n_tokens, max_cover), dtype=torch.bool, device=device)

        for pos in range(n_tokens):
            cover = self.cst.cover_set_with_depth(pos)
            for slot, (node_idx, depth) in enumerate(cover):
                if slot >= max_cover:
                    break
                node_indices[pos, slot] = node_idx
                node_depths[pos, slot] = depth
                valid_mask[pos, slot] = True

        return node_indices, node_depths, valid_mask

    def _get_cover_table(
        self, n_tokens: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if n_tokens <= self.max_seq_len:
            key = str(device)
            cached = self._cover_table_cache.get(key)
            if cached is None:
                cached = tuple(t.to(device) for t in self._cover_table_master)
                self._cover_table_cache[key] = cached
            return cached[0][:n_tokens], cached[1][:n_tokens], cached[2][:n_tokens]

        key = (n_tokens, str(device))
        cached = self._cover_table_overflow_cache.get(key)
        if cached is None:
            cached = self._build_cover_table(n_tokens, device)
            self._cover_table_overflow_cache[key] = cached
        return cached

    def _query_tree(self, q: torch.Tensor, tree: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, n_heads, dim = q.shape
        _, _, _, _, k_dim = tree.shape
        if k_dim != dim:
            raise ValueError("Tree head dim mismatch in _query_tree.")

        node_indices, node_depths, valid_mask = self._get_cover_table(seq_len, q.device)
        max_cover = node_indices.shape[1]

        node_indices_safe = node_indices.clamp(min=0)
        f_nodes = tree[:, node_indices_safe, :, :, :]  # (B, N, L, H, K, D)

        out = torch.zeros(bsz, seq_len, n_heads, dim, device=q.device, dtype=q.dtype)
        q_flat = q.reshape(bsz * seq_len, n_heads, dim)

        for depth in range(self.log_n + 1):
            depth_mask = (node_depths == depth) & valid_mask
            if not bool(depth_mask.any()):
                continue

            q_depth = self.ddq.get_query(q_flat, depth).view(bsz, seq_len, n_heads, dim)
            scale = self.ddq.get_scale(depth)
            k_depth = self._k_at_depth(depth)

            # Vectorized over all cover slots at this depth.
            f_depth = f_nodes[:, :, :, :, :k_depth, :]  # (B, N, L, H, K_d, D)
            scores = (q_depth.unsqueeze(2).unsqueeze(4) * f_depth).sum(-1) * scale
            weights = F.softmax(scores, dim=-1)
            attended = (weights.unsqueeze(-1) * f_depth).sum(-2)  # (B, N, L, H, D)
            slot_mask = depth_mask.view(1, seq_len, max_cover, 1, 1).to(attended.dtype)
            out = out + (attended * slot_mask).sum(dim=2)

        # Preserve the original behavior: mean over nodes in cover set.
        counts = valid_mask.sum(dim=1).clamp(min=1).view(1, seq_len, 1, 1).to(out.dtype)
        return (out / counts).to(q.dtype)

    def forward(self, x: torch.Tensor, mask=None, jacobi_iters: int = 0) -> torch.Tensor:
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True

        bsz, seq_len, dim = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)

        if self.tree_mode in {"fast_hybrid", "flash_only"}:
            local_out = self._flash_causal_attention(q, x)
        else:
            local_out = self._local_attention(q, x)

        if self.tree_mode in {"local_only", "flash_only"}:
            out_heads = local_out
        elif self.tree_mode == "fast_hybrid":
            global_out = self._linear_global_attention(x)
            out_heads = local_out + torch.sigmoid(self.hybrid_alpha) * global_out
        else:
            v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
            tree = self._build_tree_jacobi(v, jacobi_iters=jacobi_iters)
            tree_out = self._query_tree(q, tree)
            out_heads = local_out + tree_out

        out = out_heads.reshape(bsz, seq_len, dim)
        out = out * torch.sigmoid(self.gate_proj(x))
        out = self.dropout(self.out_proj(out))

        if squeeze:
            out = out.squeeze(0)
        return out

    def forward_incremental(
        self, x_new: torch.Tensor, cache: "DCWTv2InferenceCache"
    ) -> torch.Tensor:
        if self.tree_mode != "full":
            raise RuntimeError("Incremental cache path requires tree_mode='full'.")
        if x_new.shape[:2] != (1, 1):
            raise ValueError("forward_incremental expects x_new shape (1, 1, embedding_dim).")

        q_new = self.q_proj(x_new).view(1, self.num_heads, self.head_dim)
        v_new = self.v_proj(x_new).view(1, self.num_heads, self.head_dim)
        cache.insert(
            v_new,
            self.gated_wave_merge,
            coupling_fn=self._apply_cross_head_coupling,
        )
        out_heads = cache.query(q_new, self.ddq).unsqueeze(1)

        out = out_heads.reshape(1, 1, self.embedding_dim)
        out = out * torch.sigmoid(self.gate_proj(x_new))
        return self.dropout(self.out_proj(out))


class DCWTv2InferenceCache(nn.Module):
    """
    Optional inference cache for online generation (O(log n) insert/query).

    This module is provided for completeness and proposal parity.
    """

    def __init__(
        self,
        max_len: int,
        num_heads: int,
        head_dim: int,
        k_max: int,
        local_window: int,
        device: torch.device,
    ):
        super().__init__()
        self.max_len = max_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.k_max = k_max
        self.local_window = local_window
        self.device = device

        self.cst = CausalSegmentTree(max_len)
        self.tree: Dict[int, torch.Tensor] = {}

        self.local_kv = torch.zeros(1, local_window, num_heads, head_dim, device=device)
        self.local_write_ptr = 0
        self.n_tokens = 0

    def insert(
        self,
        v_new: torch.Tensor,  # (1, H, D)
        gated_merge_fn,
        coupling_fn=None,
    ) -> None:
        if self.n_tokens >= self.max_len:
            raise RuntimeError("Inference cache is full.")

        leaf_idx = self.cst.leaf_index(self.n_tokens)

        slot = self.local_write_ptr % self.local_window
        self.local_kv[0, slot, :, :] = v_new[0]
        self.local_write_ptr += 1

        self.tree[leaf_idx] = v_new.unsqueeze(2)  # K=1

        idx = leaf_idx >> 1
        while idx >= 1:
            left_idx = 2 * idx
            right_idx = 2 * idx + 1
            has_left = left_idx in self.tree
            has_right = right_idx in self.tree
            if not has_left and not has_right:
                break

            depth = self.cst.depth_from_leaf(idx)
            if has_left and has_right:
                node = gated_merge_fn(self.tree[left_idx], self.tree[right_idx], depth)
            elif has_left:
                node = self.tree[left_idx]
            else:
                node = self.tree[right_idx]

            if coupling_fn is not None:
                node = coupling_fn(node, depth)

            self.tree[idx] = node
            idx >>= 1

        self.n_tokens += 1

    def query(self, q_new: torch.Tensor, ddq_module: DepthDecomposedQuery) -> torch.Tensor:
        # q_new: (1, H, D)
        pos = self.n_tokens
        local_out = torch.zeros(1, self.num_heads, self.head_dim, device=self.device)
        n_local = min(pos, self.local_window)

        if n_local > 0:
            if self.local_write_ptr <= self.local_window:
                local_v = self.local_kv[0, :n_local, :, :]
            else:
                end = self.local_write_ptr % self.local_window
                indices = [(end - n_local + i) % self.local_window for i in range(n_local)]
                local_v = self.local_kv[0, indices, :, :]

            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.einsum("hd,khd->hk", q_new[0], local_v) * scale
            weights = F.softmax(scores, dim=-1)
            local_out[0] = torch.einsum("hk,khd->hd", weights, local_v)

        tree_out = torch.zeros(1, self.num_heads, self.head_dim, device=self.device)
        node_out: List[torch.Tensor] = []

        for node_idx, depth in self.cst.cover_set_with_depth(pos):
            if node_idx not in self.tree:
                continue
            f_node = self.tree[node_idx]
            q_depth = ddq_module.get_query(q_new, depth)
            scale = ddq_module.get_scale(depth)
            scores = (q_depth.unsqueeze(2) * f_node).sum(-1) * scale
            weights = F.softmax(scores, dim=-1)
            node_out.append((weights.unsqueeze(-1) * f_node).sum(2))

        if node_out:
            tree_out = torch.stack(node_out, dim=0).mean(dim=0)

        return local_out + tree_out


class DCWTv2TransformerLayer(nn.Module):
    """Single pre-norm transformer block using DCWT-v2 attention."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        ffn_dim: int,
        max_seq_len: int,
        k_max: int = 8,
        local_window: int = 32,
        tree_mode: str = "flash_only",
        dropout: float = 0.1,
        use_depth_conditioned_gwm: bool = False,
        depth_embed_dim: int = 32,
        compile_gwm: bool = True,
    ):
        super().__init__()
        self.attention = DCWTv2Attention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            k_max=k_max,
            local_window=local_window,
            tree_mode=tree_mode,
            dropout=dropout,
            use_depth_conditioned_gwm=use_depth_conditioned_gwm,
            depth_embed_dim=depth_embed_dim,
            compile_gwm=compile_gwm,
        )
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embedding_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask=None, jacobi_iters: int = 0) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.norm1(x), mask, jacobi_iters=jacobi_iters))
        x = x + self.ffn(self.norm2(x))
        return x

    def forward_incremental(
        self, x_new: torch.Tensor, cache: DCWTv2InferenceCache
    ) -> torch.Tensor:
        x_new = x_new + self.dropout(self.attention.forward_incremental(self.norm1(x_new), cache))
        x_new = x_new + self.ffn(self.norm2(x_new))
        return x_new


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_cache: int = 8192):
        super().__init__()
        pe = self._make(max_cache, dim)
        self.register_buffer("pe", pe)

    @staticmethod
    def _make(length: int, dim: int) -> torch.Tensor:
        pos = torch.arange(length).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(length, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if seq_len <= self.pe.shape[0]:
            return self.pe[:seq_len].to(device)
        return self._make(seq_len, self.pe.shape[1]).to(device)


class DCWTv2Transformer(nn.Module):
    """
    Full language model with DCWT-v2 blocks.

    Notes:
    - `field_size`, `interference_interval`, and `device` are accepted for
      backward compatibility with existing scripts, but not used by DCWT-v2.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        max_seq_len: int = 256,
        k_max: int = 8,
        local_window: int = 32,
        tree_mode: str = "flash_only",
        dropout: float = 0.1,
        use_checkpoint: bool = False,
        use_depth_conditioned_gwm: bool = False,
        depth_embed_dim: int = 32,
        compile_gwm: bool = True,
        use_haar_init: bool = False,
        d_model: Optional[int] = None,
        n_layers: Optional[int] = None,
        n_heads: Optional[int] = None,
        field_size: Optional[int] = None,
        interference_interval: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        del field_size, interference_interval, device

        if d_model is not None:
            embedding_dim = d_model
        if n_layers is not None:
            num_layers = n_layers
        if n_heads is not None:
            num_heads = n_heads

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.use_checkpoint = use_checkpoint
        self.use_depth_conditioned_gwm = use_depth_conditioned_gwm

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = SinusoidalPositionalEncoding(embedding_dim, max_seq_len * 2)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                DCWTv2TransformerLayer(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    max_seq_len=max_seq_len,
                    k_max=k_max,
                    local_window=local_window,
                    tree_mode=tree_mode,
                    dropout=dropout,
                    use_depth_conditioned_gwm=use_depth_conditioned_gwm,
                    depth_embed_dim=depth_embed_dim,
                    compile_gwm=compile_gwm,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.output_projection.weight = self.token_embedding.weight

        self._init_weights()
        if use_haar_init:
            self.apply_haar_init()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def apply_haar_init(self) -> None:
        from .bio_init import init_all_depths_haar

        init_all_depths_haar(self)

    def get_tree_grad_norms(self) -> Dict[int, float]:
        agg: Dict[int, float] = {}
        cnt: Dict[int, int] = {}
        for layer in self.layers:
            layer_norms = layer.attention.get_tree_grad_norms()
            for d, v in layer_norms.items():
                agg[d] = agg.get(d, 0.0) + float(v)
                cnt[d] = cnt.get(d, 0) + 1
        return {d: agg[d] / max(cnt[d], 1) for d in agg}

    def forward(
        self,
        input_ids: torch.Tensor,
        labels=None,
        mask=None,
        jacobi_iters: int = 0,
    ):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        _, seq_len = input_ids.shape
        x = self.token_embedding(input_ids)
        x = x + self.pos_encoding(seq_len, input_ids.device).unsqueeze(0)
        x = self.dropout(x)

        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    lambda y: layer(y, mask, jacobi_iters=jacobi_iters),
                    x,
                    use_reentrant=False,
                )
            else:
                x = layer(x, mask, jacobi_iters=jacobi_iters)

        x = self.norm(x)
        logits = self.output_projection(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        generated: torch.Tensor,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        repetition_penalty: float,
    ) -> torch.Tensor:
        logits = logits / max(temperature, 1e-6)

        if repetition_penalty != 1.0 and generated.numel() > 0:
            for token_id in set(generated[0, -50:].tolist()):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            threshold = torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_remove = cumulative_probs > top_p
            sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
            sorted_remove[..., 0] = False
            remove_mask = torch.zeros_like(logits, dtype=torch.bool)
            remove_mask.scatter_(1, sorted_indices, sorted_remove)
            logits = logits.masked_fill(remove_mask, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def _generate_full_sequence(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        repetition_penalty: float,
        jacobi_iters: int = 0,
    ) -> torch.Tensor:
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            if generated.shape[1] >= self.max_seq_len:
                break
            logits, _ = self.forward(generated, jacobi_iters=jacobi_iters)
            next_token = self._sample_next_token(
                logits[:, -1, :],
                generated,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
            )
            generated = torch.cat([generated, next_token], dim=1)
        return generated

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.2,
        jacobi_iters: int = 0,
    ) -> torch.Tensor:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.shape[0] != 1:
            raise ValueError("generate currently supports batch size 1.")
        if input_ids.shape[1] == 0:
            raise ValueError("input_ids must contain at least one token.")

        if input_ids.shape[1] > self.max_seq_len:
            input_ids = input_ids[:, -self.max_seq_len:]

        all_full_mode = all(layer.attention.tree_mode == "full" for layer in self.layers)
        was_training = self.training
        self.eval()
        try:
            if not all_full_mode:
                return self._generate_full_sequence(
                    input_ids,
                    max_new_tokens,
                    temperature,
                    top_k,
                    top_p,
                    repetition_penalty,
                    jacobi_iters=jacobi_iters,
                )

            generated = input_ids.clone()
            device = generated.device
            caches = [
                DCWTv2InferenceCache(
                    max_len=self.max_seq_len,
                    num_heads=layer.attention.num_heads,
                    head_dim=layer.attention.head_dim,
                    k_max=layer.attention.k_max,
                    local_window=layer.attention.local_window,
                    device=device,
                )
                for layer in self.layers
            ]

            max_total_len = min(self.max_seq_len, generated.shape[1] + max_new_tokens)
            pos_table = self.pos_encoding(max_total_len, device)
            next_logits: Optional[torch.Tensor] = None

            for pos in range(generated.shape[1]):
                token = generated[:, pos : pos + 1]
                x = self.token_embedding(token) + pos_table[pos].view(1, 1, -1)
                x = self.dropout(x)
                for layer, cache in zip(self.layers, caches):
                    x = layer.forward_incremental(x, cache)
                x = self.norm(x)
                next_logits = self.output_projection(x)[:, 0, :]

            if next_logits is None:
                return generated

            for _ in range(max_new_tokens):
                if generated.shape[1] >= self.max_seq_len:
                    break
                next_token = self._sample_next_token(
                    next_logits,
                    generated,
                    temperature,
                    top_k,
                    top_p,
                    repetition_penalty,
                )
                generated = torch.cat([generated, next_token], dim=1)

                pos = generated.shape[1] - 1
                x = self.token_embedding(next_token) + pos_table[pos].view(1, 1, -1)
                x = self.dropout(x)
                for layer, cache in zip(self.layers, caches):
                    x = layer.forward_incremental(x, cache)
                x = self.norm(x)
                next_logits = self.output_projection(x)[:, 0, :]

            return generated
        finally:
            if was_training:
                self.train()


__all__ = [
    "k_at_depth",
    "CausalSegmentTree",
    "GatedWaveMerge",
    "DepthConditionedGWM",
    "DepthDecomposedQuery",
    "DCWTv2Attention",
    "DCWTv2InferenceCache",
    "DCWTv2TransformerLayer",
    "SinusoidalPositionalEncoding",
    "DCWTv2Transformer",
]
