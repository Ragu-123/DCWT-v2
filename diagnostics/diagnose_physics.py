"""
DCWT-v2 Diagnostics
===================
Analyzes the corrected tree-based architecture:
1. K-vector schedule by depth (MVNB)
2. Gate initialization and learned bias health (GWM)
3. DDQ depth temperatures
4. Cross-head coupling specialization
5. Causality regression
"""

import os

import torch
import torch.nn.functional as F

from src.dcwt_v2 import CausalSegmentTree, DCWTv2Transformer, k_at_depth


def _choose_checkpoint() -> str:
    candidates = [
        "wikitext2_dcwt_v2_checkpoints/best.pt",
        "bpe_dcwt_v2_checkpoints/best.pt",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No DCWT-v2 checkpoint found. Expected one of: "
        + ", ".join(candidates)
    )


def _infer_config_from_state(state: dict) -> dict:
    vocab_size, embedding_dim = state["token_embedding.weight"].shape
    layer_indices = set()
    for key in state:
        if key.startswith("layers."):
            try:
                layer_indices.add(int(key.split(".")[1]))
            except ValueError:
                continue
    num_layers = max(layer_indices) + 1 if layer_indices else 6

    num_heads = state["layers.0.attention.cross_head_coupling.0"].shape[0]
    ffn_dim = state["layers.0.ffn.0.weight"].shape[0]
    k_max = state["layers.0.attention.gated_wave_merge.parent_query_init.0"].shape[0]
    log_n = state["layers.0.attention.ddq.depth_temp"].shape[0] - 1
    max_seq_len = (2 ** log_n) - 1

    return {
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "ffn_dim": ffn_dim,
        "max_seq_len": max_seq_len,
        "k_max": k_max,
        "local_window": 32,
    }


def _build_model(config: dict, device: torch.device) -> DCWTv2Transformer:
    return DCWTv2Transformer(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        ffn_dim=config["ffn_dim"],
        max_seq_len=config["max_seq_len"],
        k_max=config["k_max"],
        local_window=config["local_window"],
        dropout=0.1,
        use_checkpoint=False,
    ).to(device)


def _print_k_schedule(max_seq_len: int, k_max: int) -> None:
    cst = CausalSegmentTree(max_seq_len)
    print(f"\n{'='*72}")
    print("  1) MVNB K-VECTOR SCHEDULE")
    print(f"{'='*72}")
    print(f"  {'Depth(leaf)':<12} {'Span':<12} {'K(depth)':<10} {'Dims/token (K*D/span, D=256)':<32}")
    print(f"  {'-'*12} {'-'*12} {'-'*10} {'-'*32}")
    for d_leaf in range(cst.log_n + 1):
        span = 2 ** d_leaf
        k = k_at_depth(d_leaf, k_max=k_max)
        dims_per_token = (k * 256) / span
        print(f"  {d_leaf:<12} {span:<12} {k:<10} {dims_per_token:<32.2f}")


def _print_layer_diagnostics(model: DCWTv2Transformer) -> None:
    print(f"\n{'='*72}")
    print("  2) LAYER DIAGNOSTICS (GWM + DDQ + COUPLING)")
    print(f"{'='*72}")
    for li, layer in enumerate(model.layers):
        attn = layer.attention
        merge = attn.gated_wave_merge
        ddq = attn.ddq

        gate_l_bias = torch.stack([mod.bias.detach() for mod in merge.gate_left]).mean().item()
        gate_r_bias = torch.stack([mod.bias.detach() for mod in merge.gate_right]).mean().item()

        temps = F.softplus(ddq.depth_temp.detach())
        temp_min = temps.min().item()
        temp_max = temps.max().item()

        entropies = []
        for depth, coupling in enumerate(attn.cross_head_coupling):
            probs = F.softmax(coupling.detach(), dim=-1)
            entropy = -(probs * (probs + 1e-12).log()).sum(dim=-1).mean().item()
            entropies.append(entropy)

        print(f"\n  Layer {li + 1}:")
        print(f"    Gate bias (left/right): {gate_l_bias:.3f} / {gate_r_bias:.3f}")
        print(f"    DDQ temp range: {temp_min:.3f} .. {temp_max:.3f}")
        print(f"    Coupling entropy mean: {sum(entropies)/len(entropies):.3f}")


def _causality_check(model: DCWTv2Transformer, device: torch.device) -> None:
    print(f"\n{'='*72}")
    print("  3) CAUSALITY CHECK")
    print(f"{'='*72}")

    model.eval()
    seq_len = 24
    x1 = torch.randint(0, model.vocab_size, (1, seq_len), device=device)
    x2 = x1.clone()
    x2[0, -1] = (x1[0, -1] + 7) % model.vocab_size

    with torch.no_grad():
        l1, _ = model(x1)
        l2, _ = model(x2)

    past_diff = (l1[0, : seq_len - 1] - l2[0, : seq_len - 1]).abs().max().item()
    changed_diff = (l1[0, -1] - l2[0, -1]).abs().max().item()

    print(f"  Max past-position diff: {past_diff:.6e}")
    print(f"  Changed-position diff:  {changed_diff:.6e}")
    verdict = "PASS" if past_diff < 1e-5 else "FAIL"
    print(f"  Verdict: {verdict}")


def main() -> None:
    print("=" * 72)
    print("  DCWT-v2 DIAGNOSTICS")
    print("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = _choose_checkpoint()
    print(f"\nUsing checkpoint: {ckpt}")
    print(f"Device: {device}")

    state = torch.load(ckpt, map_location=device, weights_only=True)
    config = _infer_config_from_state(state)
    model = _build_model(config=config, device=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    _print_k_schedule(max_seq_len=config["max_seq_len"], k_max=config["k_max"])
    _print_layer_diagnostics(model)
    _causality_check(model, device)

    print(f"\n{'='*72}")
    print("  DIAGNOSTICS COMPLETE")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
