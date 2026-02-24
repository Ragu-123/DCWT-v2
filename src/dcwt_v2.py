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

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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

        d2 = self.head_dim // 2
        f_re = f[..., :d2]
        f_im = f[..., d2:]

        pr = phi_real.view(1, self.num_heads, 1, 1)
        pi_ = phi_imag.view(1, self.num_heads, 1, 1)

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
        return parent_norm + alpha * skip


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
        dropout: float = 0.1,
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

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        self.k_local_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_local_proj = nn.Linear(embedding_dim, embedding_dim)

        self.cst = CausalSegmentTree(max_seq_len)
        self.log_n = self.cst.log_n
        self.leaf_start = self.cst.leaf_start

        self.gated_wave_merge = GatedWaveMerge(
            num_heads=num_heads,
            head_dim=self.head_dim,
            max_depth=self.log_n,
            k_max=k_max,
        )
        self.ddq = DepthDecomposedQuery(num_heads, self.head_dim, self.log_n)

        self.cross_head_coupling = nn.ParameterList(
            [nn.Parameter(torch.eye(num_heads) + 0.01 * torch.randn(num_heads, num_heads)) for _ in range(self.log_n + 1)]
        )

        self.gate_proj = nn.Linear(embedding_dim, embedding_dim)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 2.0)

        self.dropout = nn.Dropout(dropout)

    def _apply_cross_head_coupling(self, f: torch.Tensor, depth: int) -> torch.Tensor:
        coupling = F.softmax(self.cross_head_coupling[depth], dim=-1)  # (H, H)
        return torch.einsum("ij,bjkd->bikd", coupling, f)

    def _build_tree_training(self, v: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Build sparse tree from all sequence values.
        v: (B, N, H, D_head)
        """
        bsz, seq_len, _, _ = v.shape
        tree: Dict[int, torch.Tensor] = {}

        for i in range(seq_len):
            tree[self.leaf_start + i] = v[:, i, :, :].unsqueeze(2)  # K=1

        for idx in range(self.leaf_start - 1, 0, -1):
            left_idx = 2 * idx
            right_idx = 2 * idx + 1
            has_left = left_idx in tree
            has_right = right_idx in tree
            if not has_left and not has_right:
                continue

            depth = self.cst.depth_from_leaf(idx)
            if has_left and has_right:
                parent = self.gated_wave_merge(tree[left_idx], tree[right_idx], depth)
            elif has_left:
                parent = tree[left_idx]
            else:
                parent = tree[right_idx]

            tree[idx] = self._apply_cross_head_coupling(parent, depth)

        return tree

    def _local_attention(self, q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, n_heads, dim = q.shape
        window = min(self.local_window, seq_len)

        k_local = self.k_local_proj(x).view(bsz, seq_len, n_heads, dim)
        v_local = self.v_local_proj(x).view(bsz, seq_len, n_heads, dim)
        out = torch.zeros(bsz, seq_len, n_heads, dim, device=x.device, dtype=x.dtype)

        scale = 1.0 / math.sqrt(dim)
        for i in range(seq_len):
            left = max(0, i - window)
            right = i
            if left >= right:
                continue
            q_i = q[:, i, :, :]
            k_win = k_local[:, left:right, :, :]
            v_win = v_local[:, left:right, :, :]
            scores = torch.einsum("bhd,bkhd->bhk", q_i, k_win) * scale
            weights = F.softmax(scores, dim=-1)
            out[:, i, :, :] = torch.einsum("bhk,bkhd->bhd", weights, v_win)

        return out

    def _query_tree(self, q: torch.Tensor, tree: Dict[int, torch.Tensor]) -> torch.Tensor:
        bsz, seq_len, n_heads, dim = q.shape
        out = torch.zeros(bsz, seq_len, n_heads, dim, device=q.device, dtype=q.dtype)

        for i in range(seq_len):
            cover = self.cst.cover_set_with_depth(i)
            if not cover:
                continue

            q_base = q[:, i, :, :]
            node_out: List[torch.Tensor] = []
            for node_idx, depth in cover:
                if node_idx not in tree:
                    continue
                f_node = tree[node_idx]  # (B, H, K, D)
                q_depth = self.ddq.get_query(q_base, depth)  # (B, H, D)
                scale = self.ddq.get_scale(depth)
                scores = (q_depth.unsqueeze(2) * f_node).sum(-1) * scale
                weights = F.softmax(scores, dim=-1)
                node_out.append((weights.unsqueeze(-1) * f_node).sum(2))

            if node_out:
                out[:, i, :, :] = torch.stack(node_out, dim=0).mean(dim=0)

        return out

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True

        bsz, seq_len, dim = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)

        local_out = self._local_attention(q, x)
        tree = self._build_tree_training(v)
        tree_out = self._query_tree(q, tree)

        out = (local_out + tree_out).reshape(bsz, seq_len, dim)
        out = out * torch.sigmoid(self.gate_proj(x))
        out = self.dropout(self.out_proj(out))

        if squeeze:
            out = out.squeeze(0)
        return out


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
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = DCWTv2Attention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            k_max=k_max,
            local_window=local_window,
            dropout=dropout,
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

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.norm1(x), mask))
        x = x + self.ffn(self.norm2(x))
        return x


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
        dropout: float = 0.1,
        use_checkpoint: bool = False,
        field_size: Optional[int] = None,
        interference_interval: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        del field_size, interference_interval, device

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.use_checkpoint = use_checkpoint

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
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.output_projection.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels=None, mask=None):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        _, seq_len = input_ids.shape
        x = self.token_embedding(input_ids)
        x = x + self.pos_encoding(seq_len, input_ids.device).unsqueeze(0)
        x = self.dropout(x)

        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, mask, use_reentrant=False)
            else:
                x = layer(x, mask)

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


__all__ = [
    "k_at_depth",
    "CausalSegmentTree",
    "GatedWaveMerge",
    "DepthDecomposedQuery",
    "DCWTv2Attention",
    "DCWTv2InferenceCache",
    "DCWTv2TransformerLayer",
    "SinusoidalPositionalEncoding",
    "DCWTv2Transformer",
]

