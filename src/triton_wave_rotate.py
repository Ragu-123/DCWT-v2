"""
Optional Triton kernel for DCWT-v2 wave rotation.

Falls back to PyTorch path when Triton is not available.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _wave_rotate_kernel(
        f_re_ptr,
        f_im_ptr,
        out_re_ptr,
        out_im_ptr,
        phi_real_ptr,
        phi_imag_ptr,
        B,
        M,
        H,
        K,
        D2: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_outer = tl.program_id(0)
        pid_d = tl.program_id(1)

        k_idx = pid_outer % K
        tmp = pid_outer // K
        h_idx = tmp % H
        tmp = tmp // H
        m_idx = tmp % M
        b_idx = tmp // M

        phi_r = tl.load(phi_real_ptr + h_idx)
        phi_i = tl.load(phi_imag_ptr + h_idx)

        base = ((b_idx * M + m_idx) * H + h_idx) * K * D2 + k_idx * D2
        d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = d_offsets < D2

        f_re = tl.load(f_re_ptr + base + d_offsets, mask=mask, other=0.0)
        f_im = tl.load(f_im_ptr + base + d_offsets, mask=mask, other=0.0)

        out_re = phi_r * f_re - phi_i * f_im
        out_im = phi_i * f_re + phi_r * f_im

        tl.store(out_re_ptr + base + d_offsets, out_re, mask=mask)
        tl.store(out_im_ptr + base + d_offsets, out_im, mask=mask)


def wave_rotate_triton(
    f: torch.Tensor, phi_real: torch.Tensor, phi_imag: torch.Tensor
) -> torch.Tensor:
    """
    Apply complex rotation with Triton when available.

    Args:
      f: (B,H,K,D) or (B,M,H,K,D)
      phi_real: (H,)
      phi_imag: (H,)
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available.")
    if not f.is_cuda:
        raise RuntimeError("Triton rotation requires CUDA tensor input.")

    if f.dim() == 4:
        f = f.unsqueeze(1)
        squeeze_m = True
    elif f.dim() == 5:
        squeeze_m = False
    else:
        raise ValueError(f"Unsupported rank for wave_rotate_triton: {f.dim()}")

    bsz, n_nodes, n_heads, k_slots, dim = f.shape
    if dim % 2 != 0:
        raise ValueError("Head dimension must be even for complex rotation.")
    d2 = dim // 2

    f_re = f[..., :d2].contiguous()
    f_im = f[..., d2:].contiguous()
    out_re = torch.empty_like(f_re)
    out_im = torch.empty_like(f_im)

    phi_r = phi_real.contiguous().float()
    phi_i = phi_imag.contiguous().float()

    block_d = min(64, triton.next_power_of_2(d2))
    grid = (bsz * n_nodes * n_heads * k_slots, triton.cdiv(d2, block_d))

    _wave_rotate_kernel[grid](
        f_re,
        f_im,
        out_re,
        out_im,
        phi_r,
        phi_i,
        bsz,
        n_nodes,
        n_heads,
        k_slots,
        D2=d2,
        BLOCK_D=block_d,
    )

    out = torch.cat([out_re, out_im], dim=-1)
    if squeeze_m:
        out = out.squeeze(1)
    return out
