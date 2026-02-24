"""
DCWT-v2 causality regression.

Checks the critical invariant:
changing a future token must not change past-token logits.
"""

import torch

from src.dcwt_v2 import DCWTv2Transformer


def test_causality():
    print("=" * 65)
    print("  DCWT-v2 CAUSALITY TEST")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCWTv2Transformer(
        vocab_size=128,
        embedding_dim=64,
        num_layers=2,
        num_heads=4,
        ffn_dim=128,
        max_seq_len=64,
        k_max=4,
        local_window=8,
        dropout=0.0,
        use_checkpoint=False,
    ).to(device)
    model.eval()

    seq_len = 20
    a = torch.randint(0, 128, (1, seq_len), device=device)
    b = a.clone()
    b[0, -1] = (a[0, -1] + 37) % 128

    with torch.no_grad():
        logits_a, _ = model(a)
        logits_b, _ = model(b)

    max_diff_past = (logits_a[0, : seq_len - 1] - logits_b[0, : seq_len - 1]).abs().max().item()
    last_diff = (logits_a[0, -1] - logits_b[0, -1]).abs().max().item()

    print(f"  Max diff on past positions: {max_diff_past:.6e}")
    print(f"  Max diff on changed position: {last_diff:.6e}")

    if max_diff_past >= 1e-5:
        raise AssertionError(f"CAUSALITY VIOLATION: past logits changed by {max_diff_past:.6e}")

    print("  Causality check passed.")
    print("=" * 65)


if __name__ == "__main__":
    test_causality()

