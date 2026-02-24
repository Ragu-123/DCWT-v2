"""
Backward-compatible attention entrypoint.

The repository now uses DCWT-v2 as the primary architecture. This module keeps
the historical `WaveFieldAttention` import path working.
"""

from .dcwt_v2 import DCWTv2Attention


class WaveFieldAttention(DCWTv2Attention):
    """
    Compatibility alias to DCWT-v2 attention.

    Legacy arguments such as `field_size` and `device` are accepted but ignored.
    """

    def __init__(
        self,
        embedding_dim,
        num_heads,
        field_size=512,
        max_seq_len=128,
        device="cuda",
        k_max=8,
        local_window=32,
        dropout=0.1,
    ):
        del field_size, device
        super().__init__(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            k_max=k_max,
            local_window=local_window,
            dropout=dropout,
        )


__all__ = ["WaveFieldAttention", "DCWTv2Attention"]

