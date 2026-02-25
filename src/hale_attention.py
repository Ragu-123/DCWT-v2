"""
HALE: Hierarchical Adaptive Linear-Enhanced Attention.

This module provides a causal, multi-scale attention architecture with:
- local exact attention (windowed SDPA),
- causal linear attention (ELU+1 feature map),
- causal Haar-style multi-scale context,
- adaptive local/global mixing gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Causal linear attention with phi(x)=ELU(x)+1.

    q, k, v: [B, H, N, D]
    returns: [B, H, N, D]
    """
    q_phi = F.elu(q) + 1.0
    k_phi = F.elu(k) + 1.0

    kv = torch.einsum("bhnd,bhne->bhnde", k_phi, v)
    kv_prefix = kv.cumsum(dim=2)
    out = torch.einsum("bhnd,bhnde->bhne", q_phi, kv_prefix)

    k_prefix = k_phi.cumsum(dim=2)
    norm = (q_phi * k_prefix).sum(dim=-1, keepdim=True).clamp_min(eps)
    return out / norm


def causal_linear_attention_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int = 128,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Memory-efficient causal linear attention.
    """
    q_phi = F.elu(q) + 1.0
    k_phi = F.elu(k) + 1.0

    bsz, n_heads, n_tokens, dim = q_phi.shape
    out = torch.zeros_like(q_phi)
    state_kv = torch.zeros(bsz, n_heads, dim, dim, device=q.device, dtype=q.dtype)
    state_k = torch.zeros(bsz, n_heads, dim, device=q.device, dtype=q.dtype)

    for start in range(0, n_tokens, chunk_size):
        end = min(start + chunk_size, n_tokens)

        q_c = q_phi[:, :, start:end, :]
        k_c = k_phi[:, :, start:end, :]
        v_c = v[:, :, start:end, :]

        kv_c = torch.einsum("bhcd,bhce->bhcde", k_c, v_c)
        kv_scan = kv_c.cumsum(dim=2)

        out_cross = torch.einsum("bhcd,bhde->bhce", q_c, state_kv)
        out_intra = torch.einsum("bhcd,bhcde->bhce", q_c, kv_scan)

        k_scan = k_c.cumsum(dim=2)
        norm_cross = (q_c * state_k.unsqueeze(2)).sum(dim=-1, keepdim=True)
        norm_intra = (q_c * k_scan).sum(dim=-1, keepdim=True)
        out[:, :, start:end, :] = (out_cross + out_intra) / (norm_cross + norm_intra).clamp_min(eps)

        state_kv = state_kv + kv_c.sum(dim=2)
        state_k = state_k + k_c.sum(dim=2)

    return out


class CausalHaarContext(nn.Module):
    """
    Multi-scale causal context using block-causal means at each scale.
    """

    def __init__(self, head_dim: int, num_levels: int = 4):
        super().__init__()
        self.head_dim = head_dim
        self.num_levels = num_levels

        self.level_k_proj = nn.ModuleList(
            [nn.Linear(head_dim, head_dim, bias=False) for _ in range(num_levels)]
        )
        self.level_v_proj = nn.ModuleList(
            [nn.Linear(head_dim, head_dim, bias=False) for _ in range(num_levels)]
        )
        self.scale_weights = nn.Parameter(torch.zeros(num_levels))

    def _causal_block_mean(self, x: torch.Tensor, block: int) -> torch.Tensor:
        # x: [B, H, N, D]
        bsz, n_heads, n_tokens, dim = x.shape
        prefix = x.cumsum(dim=2)
        idx = torch.arange(n_tokens, device=x.device, dtype=torch.long)
        start = (idx // block) * block
        prev_idx = (start - 1).clamp(min=0)

        prev = prefix[:, :, prev_idx, :]
        start_mask = (start > 0).view(1, 1, n_tokens, 1)
        prev = torch.where(start_mask, prev, torch.zeros_like(prev))
        sums = prefix - prev

        lengths = (idx - start + 1).to(x.dtype).view(1, 1, n_tokens, 1)
        return sums / lengths

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, chunk_size: int = 128) -> torch.Tensor:
        # q,k,v: [B, H, N, D]
        outputs: List[torch.Tensor] = []
        scale_w = F.softmax(self.scale_weights, dim=0)

        for level in range(self.num_levels):
            block = 2 ** (level + 1)
            k_mean = self._causal_block_mean(k, block)
            v_mean = self._causal_block_mean(v, block)

            k_lvl = self.level_k_proj[level](k_mean)
            v_lvl = self.level_v_proj[level](v_mean)
            if q.shape[2] > chunk_size * 2:
                lvl_out = causal_linear_attention_chunked(q, k_lvl, v_lvl, chunk_size=chunk_size)
            else:
                lvl_out = causal_linear_attention(q, k_lvl, v_lvl)
            outputs.append(scale_w[level] * lvl_out)

        if not outputs:
            return torch.zeros_like(q)
        return torch.stack(outputs, dim=0).sum(dim=0)


class AdaptiveReasoningGate(nn.Module):
    """
    Learns per-token local/global mixing.
    """

    def __init__(self, model_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(2 * model_dim, model_dim)
        self.gate_out = nn.Linear(model_dim, 1)
        nn.init.zeros_(self.gate_out.weight)
        nn.init.zeros_(self.gate_out.bias)

    def forward(self, x: torch.Tensor, local_out: torch.Tensor, global_out: torch.Tensor) -> torch.Tensor:
        diff = local_out - global_out
        gate_h = F.silu(self.gate_proj(torch.cat([x, diff], dim=-1)))
        alpha = torch.sigmoid(self.gate_out(gate_h))
        return alpha * local_out + (1.0 - alpha) * global_out


@dataclass
class HALEInferenceCache:
    local_window: int
    local_k: torch.Tensor        # [B, H, W, D]
    local_v: torch.Tensor        # [B, H, W, D]
    local_write_ptr: int
    n_tokens: int
    linear_kv: torch.Tensor      # [B, H, D, D]
    linear_k: torch.Tensor       # [B, H, D]
    haar_kv: List[torch.Tensor]  # each [B, H, D, D]
    haar_k: List[torch.Tensor]   # each [B, H, D]
    haar_accum_k: List[torch.Tensor]  # each [B, H, D]
    haar_accum_v: List[torch.Tensor]  # each [B, H, D]
    haar_counts: List[int]

    @classmethod
    def init(
        cls,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        local_window: int,
        num_haar_levels: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "HALEInferenceCache":
        return cls(
            local_window=local_window,
            local_k=torch.zeros(batch_size, num_heads, local_window, head_dim, device=device, dtype=dtype),
            local_v=torch.zeros(batch_size, num_heads, local_window, head_dim, device=device, dtype=dtype),
            local_write_ptr=0,
            n_tokens=0,
            linear_kv=torch.zeros(batch_size, num_heads, head_dim, head_dim, device=device, dtype=dtype),
            linear_k=torch.zeros(batch_size, num_heads, head_dim, device=device, dtype=dtype),
            haar_kv=[
                torch.zeros(batch_size, num_heads, head_dim, head_dim, device=device, dtype=dtype)
                for _ in range(num_haar_levels)
            ],
            haar_k=[
                torch.zeros(batch_size, num_heads, head_dim, device=device, dtype=dtype)
                for _ in range(num_haar_levels)
            ],
            haar_accum_k=[
                torch.zeros(batch_size, num_heads, head_dim, device=device, dtype=dtype)
                for _ in range(num_haar_levels)
            ],
            haar_accum_v=[
                torch.zeros(batch_size, num_heads, head_dim, device=device, dtype=dtype)
                for _ in range(num_haar_levels)
            ],
            haar_counts=[0 for _ in range(num_haar_levels)],
        )


class HALEAttention(nn.Module):
    """
    Hierarchical Adaptive Linear-Enhanced Attention.

    API-compatible with DCWTv2Attention constructor arguments.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        max_seq_len: int = 2048,
        k_max: int = 8,
        local_window: int = 64,
        tree_mode: str = "hale",
        dropout: float = 0.1,
        num_haar_levels: int = 4,
        chunk_size: int = 128,
        use_depth_conditioned_gwm: bool = False,
        depth_embed_dim: int = 32,
        compile_gwm: bool = False,
    ):
        super().__init__()
        del max_seq_len, k_max, tree_mode, use_depth_conditioned_gwm, depth_embed_dim, compile_gwm
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.local_window = local_window
        self.chunk_size = chunk_size
        self.num_haar_levels = num_haar_levels

        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.k_local_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.v_local_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.haar_context = CausalHaarContext(self.head_dim, num_levels=num_haar_levels)
        self.arg_gate = AdaptiveReasoningGate(embedding_dim)

        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self._local_mask_cache: dict[Tuple[int, int, str], torch.Tensor] = {}

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_tokens, dim = x.shape
        return x.view(bsz, n_tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_heads, n_tokens, dim = x.shape
        return x.transpose(1, 2).reshape(bsz, n_tokens, n_heads * dim)

    def _get_local_mask(self, n_tokens: int, window: int, device: torch.device) -> torch.Tensor:
        key = (n_tokens, window, str(device))
        cached = self._local_mask_cache.get(key)
        if cached is not None:
            return cached
        rows = torch.arange(n_tokens, device=device).unsqueeze(1)
        cols = torch.arange(n_tokens, device=device).unsqueeze(0)
        mask = (cols <= rows) & (cols > rows - window)
        self._local_mask_cache[key] = mask
        return mask

    def _local_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        _, _, n_tokens, dim = q.shape
        window = min(self.local_window, n_tokens)
        mask = self._get_local_mask(n_tokens, window, q.device)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            scale=1.0 / math.sqrt(dim),
        )
        return torch.nan_to_num(out, nan=0.0)

    def _global_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if q.shape[2] > self.chunk_size * 2:
            linear_out = causal_linear_attention_chunked(q, k, v, chunk_size=self.chunk_size)
        else:
            linear_out = causal_linear_attention(q, k, v)
        haar_out = self.haar_context(q, k, v, chunk_size=self.chunk_size)
        return linear_out + haar_out

    def forward(self, x: torch.Tensor, mask=None, jacobi_iters: int = 0) -> torch.Tensor:
        del mask, jacobi_iters
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True

        q = self._reshape_heads(self.q_proj(x))
        k = self._reshape_heads(self.k_proj(x))
        v = self._reshape_heads(self.v_proj(x))
        k_local = self._reshape_heads(self.k_local_proj(x))
        v_local = self._reshape_heads(self.v_local_proj(x))

        local_out = self._local_attention(q, k_local, v_local)
        global_out = self._global_attention(q, k, v)

        local_merged = self._merge_heads(local_out)
        global_merged = self._merge_heads(global_out)
        mixed = self.arg_gate(x, local_merged, global_merged)

        out = self.dropout(self.out_proj(mixed))
        if squeeze:
            out = out.squeeze(0)
        return out

    def init_inference_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> HALEInferenceCache:
        return HALEInferenceCache.init(
            batch_size=batch_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            local_window=self.local_window,
            num_haar_levels=self.num_haar_levels,
            device=device,
            dtype=dtype,
        )

    def forward_incremental(self, x_new: torch.Tensor, cache: HALEInferenceCache) -> torch.Tensor:
        # x_new: [B, 1, D]
        if x_new.dim() != 3 or x_new.shape[1] != 1:
            raise ValueError("forward_incremental expects shape [B, 1, D].")

        bsz = x_new.shape[0]
        q = self._reshape_heads(self.q_proj(x_new))[:, :, 0, :]          # [B,H,D]
        k = self._reshape_heads(self.k_proj(x_new))[:, :, 0, :]
        v = self._reshape_heads(self.v_proj(x_new))[:, :, 0, :]
        k_local = self._reshape_heads(self.k_local_proj(x_new))[:, :, 0, :]
        v_local = self._reshape_heads(self.v_local_proj(x_new))[:, :, 0, :]

        # Update local ring buffer.
        slot = cache.local_write_ptr % cache.local_window
        cache.local_k[:, :, slot, :] = k_local
        cache.local_v[:, :, slot, :] = v_local
        cache.local_write_ptr += 1

        n_local = min(cache.local_write_ptr, cache.local_window)
        if cache.local_write_ptr <= cache.local_window:
            local_k_hist = cache.local_k[:, :, :n_local, :]
            local_v_hist = cache.local_v[:, :, :n_local, :]
        else:
            end = cache.local_write_ptr % cache.local_window
            idx = [(end - n_local + i) % cache.local_window for i in range(n_local)]
            local_k_hist = cache.local_k[:, :, idx, :]
            local_v_hist = cache.local_v[:, :, idx, :]

        local_scores = torch.einsum("bhd,bhkd->bhk", q, local_k_hist) / math.sqrt(self.head_dim)
        local_weights = F.softmax(local_scores, dim=-1)
        local_out = torch.einsum("bhk,bhkd->bhd", local_weights, local_v_hist)

        q_phi = F.elu(q) + 1.0
        k_phi = F.elu(k) + 1.0
        cache.linear_kv = cache.linear_kv + torch.einsum("bhd,bhe->bhde", k_phi, v)
        cache.linear_k = cache.linear_k + k_phi

        linear_out = torch.einsum("bhd,bhde->bhe", q_phi, cache.linear_kv)
        linear_norm = (q_phi * cache.linear_k).sum(dim=-1, keepdim=True).clamp_min(1e-6)
        linear_out = linear_out / linear_norm

        scale_w = F.softmax(self.haar_context.scale_weights, dim=0)
        haar_out = torch.zeros_like(linear_out)
        for level in range(self.num_haar_levels):
            k_lvl = self.haar_context.level_k_proj[level](k)
            v_lvl = self.haar_context.level_v_proj[level](v)

            cache.haar_accum_k[level] = cache.haar_accum_k[level] + k_lvl
            cache.haar_accum_v[level] = cache.haar_accum_v[level] + v_lvl
            cache.haar_counts[level] += 1

            c = float(cache.haar_counts[level])
            partial_k = cache.haar_accum_k[level] / c
            partial_v = cache.haar_accum_v[level] / c

            kv_eff = cache.haar_kv[level] + torch.einsum("bhd,bhe->bhde", partial_k, partial_v)
            k_eff = cache.haar_k[level] + partial_k

            lvl_out = torch.einsum("bhd,bhde->bhe", q_phi, kv_eff)
            lvl_norm = (q_phi * k_eff).sum(dim=-1, keepdim=True).clamp_min(1e-6)
            haar_out = haar_out + scale_w[level] * (lvl_out / lvl_norm)

            block = 2 ** (level + 1)
            if cache.haar_counts[level] >= block:
                cache.haar_kv[level] = cache.haar_kv[level] + torch.einsum(
                    "bhd,bhe->bhde", partial_k, partial_v
                )
                cache.haar_k[level] = cache.haar_k[level] + partial_k
                cache.haar_accum_k[level].zero_()
                cache.haar_accum_v[level].zero_()
                cache.haar_counts[level] = 0

        global_out = linear_out + haar_out
        local_merged = local_out.reshape(bsz, 1, self.embedding_dim)
        global_merged = global_out.reshape(bsz, 1, self.embedding_dim)

        mixed = self.arg_gate(x_new, local_merged, global_merged)
        out = self.dropout(self.out_proj(mixed))
        cache.n_tokens += 1
        return out


class HALETransformerLayer(nn.Module):
    """
    Single transformer block with HALE attention.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        ffn_dim: int,
        max_seq_len: int,
        local_window: int = 64,
        num_haar_levels: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = HALEAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            local_window=local_window,
            num_haar_levels=num_haar_levels,
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

    def forward(self, x: torch.Tensor, mask=None, jacobi_iters: int = 0) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.norm1(x), mask=mask, jacobi_iters=jacobi_iters))
        x = x + self.ffn(self.norm2(x))
        return x

    def forward_incremental(self, x_new: torch.Tensor, cache: HALEInferenceCache) -> torch.Tensor:
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


class HALETransformer(nn.Module):
    """
    Full language model with HALE blocks.
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
        local_window: int = 64,
        tree_mode: str = "hale",
        dropout: float = 0.1,
        use_checkpoint: bool = False,
        num_haar_levels: int = 4,
        chunk_size: int = 128,
        d_model: Optional[int] = None,
        n_layers: Optional[int] = None,
        n_heads: Optional[int] = None,
        field_size: Optional[int] = None,
        interference_interval: Optional[int] = None,
        device: Optional[torch.device] = None,
        use_depth_conditioned_gwm: bool = False,
        depth_embed_dim: int = 32,
        compile_gwm: bool = False,
        use_haar_init: bool = False,
    ):
        super().__init__()
        del k_max, tree_mode, field_size, interference_interval, device
        del use_depth_conditioned_gwm, depth_embed_dim, compile_gwm, use_haar_init

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
        self.chunk_size = chunk_size

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = SinusoidalPositionalEncoding(embedding_dim, max_seq_len * 2)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                HALETransformerLayer(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    max_seq_len=max_seq_len,
                    local_window=local_window,
                    num_haar_levels=num_haar_levels,
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

    def forward(
        self,
        input_ids: torch.Tensor,
        labels=None,
        mask=None,
        jacobi_iters: int = 0,
        targets=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        del jacobi_iters
        if targets is not None and labels is None:
            labels = targets
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        _, seq_len = input_ids.shape
        x = self.token_embedding(input_ids)
        x = x + self.pos_encoding(seq_len, input_ids.device).unsqueeze(0)
        x = self.dropout(x)

        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    lambda y: layer(y, mask),
                    x,
                    use_reentrant=False,
                )
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
        del jacobi_iters
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.shape[0] != 1:
            raise ValueError("generate currently supports batch size 1.")
        if input_ids.shape[1] == 0:
            raise ValueError("input_ids must contain at least one token.")

        if input_ids.shape[1] > self.max_seq_len:
            input_ids = input_ids[:, -self.max_seq_len:]

        was_training = self.training
        self.eval()
        try:
            generated = input_ids.clone()
            device = generated.device

            caches = [
                layer.attention.init_inference_cache(
                    batch_size=1, device=device, dtype=self.token_embedding.weight.dtype
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


# Backward-compatible aliases for migration scripts.
WaveFieldAttentionHALE = HALEAttention
DCWTv2AttentionHALE = HALEAttention
DCWTv2TransformerHALE = HALETransformer


__all__ = [
    "causal_linear_attention",
    "causal_linear_attention_chunked",
    "CausalHaarContext",
    "AdaptiveReasoningGate",
    "HALEInferenceCache",
    "HALEAttention",
    "HALETransformerLayer",
    "SinusoidalPositionalEncoding",
    "HALETransformer",
    "WaveFieldAttentionHALE",
    "DCWTv2AttentionHALE",
    "DCWTv2TransformerHALE",
]
