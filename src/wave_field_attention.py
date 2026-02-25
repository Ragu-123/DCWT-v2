"""
Backward-compatible attention entrypoint.

The repository now uses HALE as the primary architecture while preserving
historical import paths.
"""

from .dcwt_v2 import DCWTv2Attention
from .hale_attention import HALEAttention


class WaveFieldAttention(HALEAttention):
    """
    Compatibility alias to HALE attention.
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
        num_haar_levels=4,
        chunk_size=128,
    ):
        del field_size, device
        super().__init__(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            local_window=local_window,
            dropout=dropout,
            num_haar_levels=num_haar_levels,
            chunk_size=chunk_size,
        )


__all__ = ["WaveFieldAttention", "HALEAttention", "DCWTv2Attention"]
