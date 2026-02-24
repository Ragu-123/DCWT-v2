"""
Backward-compatible transformer entrypoint.

The project now defaults to DCWT-v2. This module preserves historical
`WaveFieldTransformer` imports and call signatures used by existing scripts.
"""

import torch.nn as nn

from .dcwt_v2 import (
    DCWTv2Attention,
    DCWTv2InferenceCache,
    DCWTv2Transformer,
    DCWTv2TransformerLayer,
    DepthDecomposedQuery,
    GatedWaveMerge,
    SinusoidalPositionalEncoding,
    k_at_depth,
)


class FieldInterferenceModule(nn.Module):
    """
    Legacy compatibility shim.

    Interference routing is replaced by the DCWT-v2 tree mechanisms. This class
    remains as an identity module so older imports do not fail.
    """

    def __init__(self, embedding_dim, dropout=0.1, initial_temperature=-2.0):
        super().__init__()
        del embedding_dim, dropout, initial_temperature

    def forward(self, x):
        return x


class WaveFieldTransformer(DCWTv2Transformer):
    """
    Compatibility alias to DCWT-v2 transformer.

    Keeps old keyword arguments (`field_size`, `interference_interval`, `device`)
    accepted for scripts that were written for Wave Field V3.x.
    """

    def __init__(
        self,
        vocab_size=50257,
        embedding_dim=256,
        num_layers=6,
        num_heads=8,
        ffn_dim=1024,
        field_size=512,
        max_seq_len=2048,
        dropout=0.1,
        use_checkpoint=False,
        interference_interval=3,
        device=None,
        k_max=8,
        local_window=32,
    ):
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            max_seq_len=max_seq_len,
            k_max=k_max,
            local_window=local_window,
            dropout=dropout,
            use_checkpoint=use_checkpoint,
            field_size=field_size,
            interference_interval=interference_interval,
            device=device,
        )


__all__ = [
    "WaveFieldTransformer",
    "FieldInterferenceModule",
    "DCWTv2Attention",
    "DCWTv2Transformer",
    "DCWTv2TransformerLayer",
    "DCWTv2InferenceCache",
    "DepthDecomposedQuery",
    "GatedWaveMerge",
    "SinusoidalPositionalEncoding",
    "k_at_depth",
]

