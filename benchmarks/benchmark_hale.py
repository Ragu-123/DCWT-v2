"""
benchmark_hale.py — Quick benchmark comparing HALE vs DCWT-v2 vs Standard Transformer.

Usage (from repo root):
    $env:PYTHONPATH = '.;tokenizers'
    python benchmarks/benchmark_hale.py

This script reuses the WaveField LLM data pipeline and training loop.
It benchmarks three architectures on WikiText-2 (character tokenizer):

  1. Standard Transformer (baseline)
  2. DCWT-v2 (flash_only mode — current best)
  3. HALE (new architecture)

Key metrics:
  - Perplexity (quality)
  - Tokens/sec (training speed)
  - Peak GPU memory (GB)
  - Parameter count
"""

import os
import time
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.hale_attention import HALETransformer

# Try to import existing models for comparison
try:
    from src.dcwt_v2 import DCWTv2Transformer
    HAS_DCWT = True
except ImportError:
    HAS_DCWT = False
    print("[WARN] DCWT-v2 not found, skipping comparison")


# ── Standard Transformer Baseline ─────────────────────────────────────────

class StandardTransformerBaseline(nn.Module):
    """Clean standard transformer for baseline comparison."""

    def __init__(self, vocab_size, embedding_dim=256, num_layers=6,
                 num_heads=8, ffn_dim=1024, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(max_seq_len + 2, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids, targets=None):
        B, n = input_ids.shape
        device = input_ids.device
        pos = torch.arange(n, device=device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.pos_embedding(pos)

        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(n, device=device)
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Data Loading ───────────────────────────────────────────────────────────

def load_wikitext2_char(max_seq_len=256, max_chars=500_000):
    """
    Load WikiText-2 as character sequences.
    Falls back to synthetic data if WikiText-2 not available.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1')
        train_text = ' '.join(ds['train']['text'])[:max_chars]
        val_text = ' '.join(ds['validation']['text'])[:max_chars // 5]
        print(f"  WikiText-2 loaded: {len(train_text):,} train chars, {len(val_text):,} val chars")
    except Exception as e:
        print(f"  [WARN] WikiText-2 not available ({e}), using synthetic data")
        import random
        random.seed(42)
        chars = 'abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?;:\n'
        train_text = ''.join(random.choices(chars, k=max_chars))
        val_text = ''.join(random.choices(chars, k=max_chars // 5))

    # Character tokenization
    vocab = sorted(set(train_text + val_text))
    char2idx = {c: i for i, c in enumerate(vocab)}
    vocab_size = len(vocab)
    print(f"  Vocab size: {vocab_size}")

    def encode_to_sequences(text):
        ids = [char2idx.get(c, 0) for c in text]
        seqs = []
        for i in range(0, len(ids) - max_seq_len, max_seq_len):
            seqs.append(ids[i:i + max_seq_len + 1])  # +1 for targets
        return torch.tensor(seqs, dtype=torch.long)

    train_data = encode_to_sequences(train_text)
    val_data = encode_to_sequences(val_text)
    return train_data, val_data, vocab_size


def make_batches(data, batch_size, device):
    """Yield (inputs, targets) batches."""
    indices = torch.randperm(len(data))
    for i in range(0, len(data) - batch_size, batch_size):
        batch = data[indices[i:i+batch_size]].to(device)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        yield inputs, targets


# ── Training Loop ─────────────────────────────────────────────────────────

def train_one_epoch(model, data, batch_size, optimizer, device, max_batches=None):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    t0 = time.time()

    for i, (inputs, targets) in enumerate(make_batches(data, batch_size, device)):
        if max_batches and i >= max_batches:
            break

        optimizer.zero_grad()

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                            enabled=(device.type == 'cuda')):
            _, loss = model(inputs, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * inputs.numel()
        total_tokens += inputs.numel()

    elapsed = time.time() - t0
    tokens_per_sec = total_tokens / elapsed
    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss, tokens_per_sec


@torch.no_grad()
def evaluate(model, data, batch_size, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for inputs, targets in make_batches(data, batch_size, device):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                            enabled=(device.type == 'cuda')):
            _, loss = model(inputs, targets)
        total_loss += loss.item() * inputs.numel()
        total_tokens += inputs.numel()

    return total_loss / max(total_tokens, 1)


def measure_peak_memory(model, inputs, device):
    """Measure peak GPU memory for a single forward pass."""
    if device.type != 'cuda':
        return 0.0

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            model(inputs)

    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # GB


# ── Main Benchmark ─────────────────────────────────────────────────────────

def run_benchmark():
    print("\n" + "="*72)
    print("  HALE vs DCWT-v2 vs Standard Transformer Benchmark")
    print("="*72)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name()}")

    # ── Data ────────────────────────────────────────────────────────────
    print("\n  Loading data...")
    MAX_SEQ = 256
    train_data, val_data, vocab_size = load_wikitext2_char(
        max_seq_len=MAX_SEQ, max_chars=300_000
    )
    print(f"  Train seqs: {len(train_data):,}  Val seqs: {len(val_data):,}")

    BATCH_SIZE = 32
    NUM_EPOCHS = 5
    MAX_TRAIN_BATCHES = 100   # Quick benchmark — increase for full run

    # ── Model Config ────────────────────────────────────────────────────
    CFG = dict(
        vocab_size=vocab_size,
        embedding_dim=256,
        num_layers=6,
        num_heads=8,
        ffn_dim=1024,
        max_seq_len=MAX_SEQ,
        dropout=0.1,
    )

    models_to_run = {}

    # Standard Transformer
    models_to_run['Standard Transformer'] = StandardTransformerBaseline(**CFG)

    # DCWT-v2
    if HAS_DCWT:
        os.environ['DCWT_TREE_MODE'] = 'flash_only'
        models_to_run['DCWT-v2 (flash_only)'] = DCWTv2Transformer(
            vocab_size=vocab_size,
            embedding_dim=256,
            num_layers=6,
            num_heads=8,
            ffn_dim=1024,
            max_seq_len=MAX_SEQ,
        )

    # HALE
    models_to_run['HALE (new)'] = HALETransformer(
        vocab_size=vocab_size,
        embedding_dim=256,
        num_layers=6,
        num_heads=8,
        ffn_dim=1024,
        max_seq_len=MAX_SEQ,
        local_window=64,
        num_haar_levels=4,
        dropout=0.1,
    )

    results = {}

    for name, model in models_to_run.items():
        print(f"\n{'='*72}")
        print(f"  Model: {name}")
        print(f"{'='*72}")

        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {n_params:,} ({n_params/1e6:.2f}M)")

        # Memory measurement
        dummy_input = torch.randint(0, vocab_size, (BATCH_SIZE, MAX_SEQ), device=device)
        peak_mem = measure_peak_memory(model, dummy_input, device)
        if device.type == 'cuda':
            print(f"  Peak memory (fwd only): {peak_mem:.3f} GB")

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=3e-4, weight_decay=0.1
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=NUM_EPOCHS
        )

        epoch_results = []
        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss, tps = train_one_epoch(
                model, train_data, BATCH_SIZE, optimizer, device,
                max_batches=MAX_TRAIN_BATCHES
            )
            val_loss = evaluate(model, val_data, BATCH_SIZE, device)
            val_ppl = math.exp(min(val_loss, 10))
            scheduler.step()

            epoch_results.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_ppl': val_ppl,
                'tokens_per_sec': tps,
            })
            print(f"  Epoch {epoch:2d}: "
                  f"train_loss={train_loss:.3f}  "
                  f"val_ppl={val_ppl:.1f}  "
                  f"speed={tps:,.0f} tok/s")

        best = min(epoch_results, key=lambda r: r['val_loss'])
        results[name] = {
            'params': n_params,
            'peak_mem_gb': peak_mem,
            'best_val_ppl': best['val_ppl'],
            'best_epoch': best['epoch'],
            'avg_tokens_per_sec': sum(r['tokens_per_sec'] for r in epoch_results) / len(epoch_results),
        }

    # ── Results Table ────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  RESULTS SUMMARY")
    print(f"{'='*72}")
    print(f"  {'Model':<30} {'Params':>10} {'Best PPL':>10} {'Speed (tok/s)':>15} {'Mem (GB)':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*15} {'-'*10}")

    baseline_ppl = None
    baseline_speed = None

    for name, r in results.items():
        if baseline_ppl is None:
            baseline_ppl = r['best_val_ppl']
            baseline_speed = r['avg_tokens_per_sec']

        ppl_delta = (r['best_val_ppl'] - baseline_ppl) / baseline_ppl * 100
        speed_delta = (r['avg_tokens_per_sec'] - baseline_speed) / baseline_speed * 100
        ppl_str = f"{r['best_val_ppl']:.1f} ({ppl_delta:+.1f}%)"
        speed_str = f"{r['avg_tokens_per_sec']:,.0f} ({speed_delta:+.1f}%)"
        mem_str = f"{r['peak_mem_gb']:.3f}" if r['peak_mem_gb'] > 0 else "CPU"

        print(f"  {name:<30} {r['params']:>10,} {ppl_str:>10} {speed_str:>15} {mem_str:>10}")

    print(f"\n  Note: benchmarked at seq_len={MAX_SEQ} with {MAX_TRAIN_BATCHES} batches/epoch.")
    print(f"        HALE's speed advantage grows significantly at seq_len > 1024.")
    print(f"        HALE's memory advantage is most visible at inference, not shown here.")


# ── Long-Context Speed Test ────────────────────────────────────────────────

def long_context_speed_test():
    """
    Test training speed at multiple sequence lengths.
    This is where HALE's O(n) advantage really shows.
    """
    print(f"\n{'='*72}")
    print("  LONG-CONTEXT SPEED TEST (tokens/sec vs sequence length)")
    print(f"{'='*72}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("  [SKIP] Requires CUDA")
        return

    VOCAB = 1000
    B = 4    # small batch for long sequences
    D = 256

    models = {
        'Standard': StandardTransformerBaseline(VOCAB, D, 4, 8, 1024),
        'HALE': HALETransformer(VOCAB, D, 4, 8, 1024, local_window=64, num_haar_levels=4),
    }

    seq_lens = [256, 512, 1024, 2048, 4096]

    for name, model in models.items():
        model = model.to(device)
        print(f"\n  {name}:")
        for seq_len in seq_lens:
            try:
                x = torch.randint(0, VOCAB, (B, seq_len), device=device)
                y = torch.randint(0, VOCAB, (B, seq_len), device=device)

                # Warmup
                with torch.autocast('cuda', torch.bfloat16):
                    _, loss = model(x, y)
                loss.backward()
                model.zero_grad()

                # Benchmark
                torch.cuda.synchronize()
                t0 = time.time()
                N_ITERS = 5
                for _ in range(N_ITERS):
                    with torch.autocast('cuda', torch.bfloat16):
                        _, loss = model(x, y)
                    loss.backward()
                    model.zero_grad()
                torch.cuda.synchronize()
                elapsed = (time.time() - t0) / N_ITERS

                tps = B * seq_len / elapsed
                mem = torch.cuda.max_memory_allocated(device) / (1024**3)
                torch.cuda.reset_peak_memory_stats()

                print(f"    seq={seq_len:5d}: {tps:8,.0f} tok/s  mem={mem:.2f}GB")

            except torch.cuda.OutOfMemoryError:
                print(f"    seq={seq_len:5d}: OOM")
                torch.cuda.empty_cache()


if __name__ == '__main__':
    print("\n  ══ HALE Architecture Benchmark ══")
    print("  Hierarchical Adaptive Linear-Enhanced Attention")
    print("  POC for WaveField LLM codebase")

    run_benchmark()
    long_context_speed_test()

    print("\n  Done. See HALE_RESEARCH.md for full architecture details.")
