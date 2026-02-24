"""
DCWT-v2 structural smoke tests.
"""

import torch

from src.dcwt_v2 import CausalSegmentTree, DCWTv2Transformer, k_at_depth


def test_forward_shape():
    model = DCWTv2Transformer(
        vocab_size=256,
        embedding_dim=128,
        num_layers=2,
        num_heads=4,
        ffn_dim=256,
        max_seq_len=64,
        k_max=4,
        local_window=8,
        dropout=0.0,
    )
    x = torch.randint(0, 256, (2, 32))
    logits, _ = model(x)
    assert logits.shape == (2, 32, 256)


def test_k_schedule():
    cst = CausalSegmentTree(64)
    for d_leaf in range(cst.log_n + 1):
        assert k_at_depth(d_leaf, k_max=4) == min(2 ** d_leaf, 4)


if __name__ == "__main__":
    test_forward_shape()
    test_k_schedule()
    print("DCWT-v2 smoke tests passed.")

