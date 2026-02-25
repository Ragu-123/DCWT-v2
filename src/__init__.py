"""
Wave Field LLM / DCWT-v2 package exports.
"""

__version__ = "4.0.0"

from .causal_field_attention import CausalFieldAttentionV2
from .causal_field_transformer import CausalFieldTransformer
from .dcwt_v2 import (
    CausalSegmentTree,
    DCWTv2Attention,
    DCWTv2InferenceCache,
    DCWTv2Transformer,
    DCWTv2TransformerLayer,
    DepthConditionedGWM,
    DepthDecomposedQuery,
    GatedWaveMerge,
    SinusoidalPositionalEncoding,
    k_at_depth,
)
from .bio_init import init_all_depths_haar, init_gwm_haar
from .bio_scheduler import SlimeMoldKScheduler, heartbeat_schedule, set_gwm_depth_frozen
from .global_context import GlobalContextModule
from .wave_field_attention import WaveFieldAttention
from .wave_field_transformer import FieldInterferenceModule, WaveFieldTransformer

__all__ = [
    "CausalFieldAttentionV2",
    "CausalFieldTransformer",
    "GlobalContextModule",
    "WaveFieldAttention",
    "WaveFieldTransformer",
    "FieldInterferenceModule",
    "CausalSegmentTree",
    "DCWTv2Attention",
    "DCWTv2InferenceCache",
    "DCWTv2Transformer",
    "DCWTv2TransformerLayer",
    "DepthConditionedGWM",
    "DepthDecomposedQuery",
    "GatedWaveMerge",
    "SinusoidalPositionalEncoding",
    "k_at_depth",
    "init_gwm_haar",
    "init_all_depths_haar",
    "set_gwm_depth_frozen",
    "heartbeat_schedule",
    "SlimeMoldKScheduler",
]
