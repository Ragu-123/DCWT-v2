"""
DCWT-v2 + BPE diagnostics.

Focus:
1. BPE sequence statistics
2. Local-window coverage pressure
3. MVNB depth capacity profile
4. DDQ temperature profile
"""

import os

import torch
import torch.nn.functional as F

from src.dcwt_v2 import CausalSegmentTree, DCWTv2Transformer, k_at_depth


def train_bpe_tokenizer(train_texts, vocab_size=8000):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        min_frequency=2,
    )
    tokenizer.train_from_iterator(train_texts, trainer=trainer)
    return tokenizer


class BPEWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def vocab_size_actual(self):
        return self.tokenizer.get_vocab_size()


def _pick_checkpoint() -> str:
    candidates = [
        "bpe_dcwt_v2_checkpoints/best.pt",
        "100m_dcwt_v2_checkpoints/best.pt",
        "wikitext2_dcwt_v2_checkpoints/best.pt",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No DCWT-v2 checkpoint found. Expected one of: "
        + ", ".join(candidates)
    )


def _token_length_stats(tok: BPEWrapper, lines):
    lengths = [len(tok.encode(line)) for line in lines if line.strip()]
    if not lengths:
        return 0.0, 0.0, 0
    mean_len = float(sum(lengths) / len(lengths))
    sorted_l = sorted(lengths)
    p95 = float(sorted_l[int(0.95 * (len(sorted_l) - 1))])
    max_len = int(sorted_l[-1])
    return mean_len, p95, max_len


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
    }


def _print_capacity_table(max_seq_len: int, k_max: int, embed_dim: int):
    cst = CausalSegmentTree(max_seq_len)
    print(f"\n{'='*70}")
    print("  3) MVNB CAPACITY PROFILE")
    print(f"{'='*70}")
    print(f"  {'Depth':<8} {'Span':<8} {'K':<8} {'Dims/token':<12}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*12}")
    for d in range(cst.log_n + 1):
        span = 2 ** d
        k = k_at_depth(d, k_max)
        dims_token = (k * embed_dim) / span
        print(f"  {d:<8} {span:<8} {k:<8} {dims_token:<12.2f}")


def main():
    print("=" * 70)
    print("  DCWT-v2 + BPE DIAGNOSTICS")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    from datasets import load_dataset

    print("\nLoading WikiText-2 and training BPE tokenizer...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_lines = [
        item["text"].strip()
        for item in ds["train"]
        if item["text"].strip() and not item["text"].strip().startswith("=")
    ]

    tok = BPEWrapper(train_bpe_tokenizer(train_lines, vocab_size=8000))
    vocab_size = tok.vocab_size_actual()
    print(f"BPE vocab size: {vocab_size}")

    mean_len, p95_len, max_len = _token_length_stats(tok, train_lines[:5000])
    print(f"Mean seq length (sample): {mean_len:.1f}")
    print(f"P95 seq length (sample):  {p95_len:.1f}")
    print(f"Max seq length (sample):  {max_len}")

    ckpt = _pick_checkpoint()
    print(f"\nLoading checkpoint: {ckpt}")
    state = torch.load(ckpt, map_location=device, weights_only=True)
    cfg = _infer_config_from_state(state)

    max_seq_len = cfg["max_seq_len"]
    local_window = 32
    k_max = cfg["k_max"]
    embed_dim = cfg["embedding_dim"]

    model = DCWTv2Transformer(
        vocab_size=cfg["vocab_size"],
        embedding_dim=cfg["embedding_dim"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        ffn_dim=cfg["ffn_dim"],
        max_seq_len=cfg["max_seq_len"],
        k_max=cfg["k_max"],
        local_window=local_window,
        dropout=0.1,
        use_checkpoint=False,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    print(f"\n{'='*70}")
    print("  1) LOCAL WINDOW PRESSURE")
    print(f"{'='*70}")
    print(f"Configured local window W: {local_window}")
    print(f"Fraction of P95 directly covered: {min(local_window / max(p95_len, 1.0), 1.0) * 100:.1f}%")
    print(f"Fraction of max directly covered: {min(local_window / max(max_len, 1), 1.0) * 100:.1f}%")

    print(f"\n{'='*70}")
    print("  2) DDQ TEMPERATURE PROFILE")
    print(f"{'='*70}")
    for i, layer in enumerate(model.layers):
        temps = F.softplus(layer.attention.ddq.depth_temp.detach())
        print(
            f"  Layer {i+1}: min={temps.min().item():.3f}, "
            f"max={temps.max().item():.3f}, mean={temps.mean().item():.3f}"
        )

    _print_capacity_table(max_seq_len=max_seq_len, k_max=k_max, embed_dim=embed_dim)

    print(f"\n{'='*70}")
    print("  DIAGNOSTICS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
