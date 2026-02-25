"""
DCWT-v2 + BPE Tokenizer — WikiText-2 Benchmark
==============================================
Byte-level BPE benchmark for Standard Transformer vs DCWT-v2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import math
import json
from torch.utils.data import DataLoader, Dataset

from src.dcwt_v2 import DCWTv2Transformer


# ======================================================================
# STANDARD TRANSFORMER BASELINE
# ======================================================================

class StandardTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, num_layers=6,
                 num_heads=8, ffn_dim=1024, max_seq_len=256, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads,
            dim_feedforward=ffn_dim, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.output_projection.weight = self.token_embedding.weight
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _generate_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    def forward(self, input_ids, labels=None, mask=None):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        B, N = input_ids.shape
        positions = torch.arange(N, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_embedding(input_ids) + self.positional_embedding(positions)
        x = self.dropout(x)
        causal_mask = self._generate_causal_mask(N, input_ids.device)
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        x = self.norm(x)
        logits = self.output_projection(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1), ignore_index=-100)
        return logits, loss


# ======================================================================
# BPE TOKENIZER
# ======================================================================

def train_bpe_tokenizer(train_texts, vocab_size=8000):
    """Train a Byte-Level BPE tokenizer on the training data."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

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
    """Wrapper to give BPE tokenizer same interface as FieldTokenizerV2."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def vocab_size_actual(self):
        return self.tokenizer.get_vocab_size()


# ======================================================================
# UTILITIES
# ======================================================================

def load_wikitext2():
    from datasets import load_dataset
    print("Loading WikiText-2...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    splits = {}
    for split_name, hf_split in [('train', 'train'), ('valid', 'validation'), ('test', 'test')]:
        lines = [item['text'].strip() for item in ds[hf_split]
                 if item['text'].strip() and not item['text'].strip().startswith('=')]
        splits[split_name] = lines
    return splits


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr = self.base_lr * (self.step_count / self.warmup_steps)
        else:
            p = (self.step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * p))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr


def encode_lines(lines, tok, max_seq_len):
    data = []
    for line in lines:
        ids = tok.encode(line)
        if len(ids) < 2:
            continue
        if len(ids) > max_seq_len:
            for s in range(0, len(ids) - 1, max_seq_len):
                c = ids[s:s + max_seq_len + 1]
                if len(c) >= 2:
                    data.append((torch.tensor(c[:-1]), torch.tensor(c[1:])))
        else:
            data.append((torch.tensor(ids[:-1]), torch.tensor(ids[1:])))
    return data


class DCWTDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    px = torch.zeros(len(xs), max_len, dtype=torch.long)
    py = torch.full((len(ys), max_len), -100, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        px[i, : x.size(0)] = x
        py[i, : y.size(0)] = y
    return px, py


def create_dataloader(data, batch_size, shuffle=True):
    num_workers = min(4, os.cpu_count() or 1)
    return DataLoader(
        DCWTDataset(data),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        collate_fn=collate_fn,
    )


def get_autocast_config(device):
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return "cuda", dtype
    return "cpu", torch.bfloat16


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    vocab_size,
    device,
    use_amp=False,
    autocast_device="cuda",
    autocast_dtype=torch.float16,
):
    model.eval()
    tl, tc, tt, n = 0, 0, 0, 0
    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(
            autocast_device, enabled=use_amp, dtype=autocast_dtype
        ):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1), ignore_index=-100)
        tl += loss.item(); n += 1
        mask = y != -100
        tc += (logits.argmax(-1)[mask] == y[mask]).sum().item()
        tt += mask.sum().item()
    model.train()
    al = tl / max(n, 1)
    return al, math.exp(min(al, 20)), tc / max(tt, 1) * 100


@torch.no_grad()
def generate_text(model, tok, seed, device, max_tokens=60,
                  temperature=0.8, top_k=50, top_p=0.92, rep_penalty=1.2):
    model.eval()
    ids = tok.encode(seed)
    if not ids:
        return seed + " [empty]"
    gen = ids.copy()
    inp = torch.tensor(ids, device=device).unsqueeze(0)
    for _ in range(max_tokens):
        logits, _ = model(inp)
        nl = logits[0, -1, :] / temperature
        for pid in set(gen[-30:]):
            if nl[pid] > 0: nl[pid] /= rep_penalty
            else: nl[pid] *= rep_penalty
        if top_k > 0:
            tv, ti = torch.topk(nl, min(top_k, nl.size(-1)))
            f = torch.full_like(nl, float('-inf'))
            f.scatter_(0, ti, tv)
        else:
            f = nl
        if top_p < 1.0:
            sl, si = torch.sort(f, descending=True)
            cp = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
            rm = cp > top_p; rm[1:] = rm[:-1].clone(); rm[0] = False
            f[si[rm]] = float('-inf')
        nid = torch.multinomial(F.softmax(f, dim=-1), 1).item()
        gen.append(nid)
        inp = torch.tensor([gen], device=device)
    model.train()
    return tok.decode(gen)


def train_model(model, train_data, val_data, tok, vocab_size, device,
                model_name, num_epochs=30, batch_size=64, grad_accum=1, peak_lr=0.0004,
                use_amp=True, autocast_device="cuda", autocast_dtype=torch.float16,
                save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)

    params = sum(p.numel() for p in model.parameters())
    print(f"\n  {model_name}: {params:,} parameters")
    print(f"  Effective batch size: {batch_size * grad_accum}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01, eps=1e-8)
    train_loader = create_dataloader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(val_data, batch_size=batch_size, shuffle=False)
    spe = max(1, len(train_loader))
    scheduler = WarmupCosineScheduler(optimizer, spe * 3, spe * num_epochs, min_lr=1e-5)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(use_amp and autocast_device == "cuda")
    )

    best_vl, best_vp, best_va, best_ep = float('inf'), float('inf'), 0, 0

    t0 = time.time()
    for epoch in range(1, num_epochs + 1):
        et = time.time()
        model.train()
        tl, nb = 0, 0
        optimizer.zero_grad(set_to_none=True)

        for step, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast(
                autocast_device, enabled=use_amp, dtype=autocast_dtype
            ):
                logits, _ = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    y.reshape(-1),
                    ignore_index=-100,
                ) / grad_accum
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            scaler.scale(loss).backward()
            tl += loss.item() * grad_accum
            nb += 1

            if step % grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        al = tl / max(nb, 1)
        et = time.time() - et

        if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
            vl, vp, va = evaluate(
                model,
                val_loader,
                vocab_size,
                device,
                use_amp=use_amp,
                autocast_device=autocast_device,
                autocast_dtype=autocast_dtype,
            )
            if vl < best_vl:
                best_vl, best_vp, best_va, best_ep = vl, vp, va, epoch
                torch.save(model.state_dict(), os.path.join(save_dir, "best.pt"))
                mk = " * BEST"
            else:
                mk = ""
            print(f"  Ep {epoch:3d}/{num_epochs} | Train {al:.4f} | Val {vl:.4f} PPL {vp:.1f} Acc {va:.1f}% | {et:.1f}s{mk}")
            if epoch % 10 == 0:
                sample = generate_text(model, tok, "The president of the", device, max_tokens=30)
                print(f"    Gen: {sample}")
        else:
            print(f"  Ep {epoch:3d}/{num_epochs} | Train {al:.4f} | {et:.1f}s")

    total = time.time() - t0
    model.load_state_dict(torch.load(os.path.join(save_dir, "best.pt"), weights_only=True))

    return {
        'model_name': model_name,
        'params': params,
        'best_ppl': best_vp,
        'best_acc': best_va,
        'best_epoch': best_ep,
        'total_time': total,
    }


# ======================================================================
# MAIN
# ======================================================================

def main():
    # Keep benchmark startup deterministic and avoid compile warmup overhead.
    os.environ["DCWT_COMPILE"] = "0"
    dcwt_tree_mode = os.environ.get("DCWT_TREE_MODE", "flash_only")

    print("=" * 65)
    print("  DCWT-v2 + BPE TOKENIZER")
    print("  Standard Transformer vs DCWT-v2 — fair BPE comparison")
    print(f"  DCWT tree_mode: {dcwt_tree_mode}")
    print("=" * 65)

    splits = load_wikitext2()

    # Train BPE tokenizer
    bpe_vocab_size = 8000
    print(f"\nTraining Byte-Level BPE tokenizer (vocab={bpe_vocab_size})...")
    raw_tokenizer = train_bpe_tokenizer(splits['train'], vocab_size=bpe_vocab_size)
    tok = BPEWrapper(raw_tokenizer)
    vocab_size = tok.vocab_size_actual()
    print(f"  BPE vocab: {vocab_size} tokens")

    # Show tokenization examples
    examples = ["The president of the united states", "Scientists discovered that"]
    for ex in examples:
        ids = tok.encode(ex)
        decoded = [tok.decode([i]) for i in ids]
        print(f"  \"{ex}\" → {len(ids)} tokens: {decoded[:10]}...")

    max_seq_len = 256
    train_data = encode_lines(splits['train'], tok, max_seq_len)
    val_data = encode_lines(splits['valid'], tok, max_seq_len)
    test_data = encode_lines(splits['test'], tok, max_seq_len)

    print(f"\n  max_seq_len: {max_seq_len}")
    print(f"  Train: {len(train_data):,} | Val: {len(val_data):,} | Test: {len(test_data):,}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = True
    autocast_device, autocast_dtype = get_autocast_config(device)
    print(
        f"  Device: {device} | AMP: {use_amp} "
        f"({autocast_device}, {str(autocast_dtype).split('.')[-1]})"
    )
    if device.type == "cuda" and hasattr(torch.backends, "cuda"):
        try:
            print(
                "  SDPA backends:"
                f" flash={torch.backends.cuda.flash_sdp_enabled()}"
                f" mem_efficient={torch.backends.cuda.mem_efficient_sdp_enabled()}"
                f" math={torch.backends.cuda.math_sdp_enabled()}"
            )
        except Exception:
            pass

    num_epochs = 30
    batch_size = 64
    grad_accum = 1
    peak_lr = 0.0004
    results = []

    # ============================================================
    # MODEL 1: Standard Transformer
    # ============================================================
    print(f"\n{'='*65}")
    print("  MODEL 1: STANDARD TRANSFORMER (O(n^2))")
    print(f"{'='*65}")

    std_model = StandardTransformer(
        vocab_size=vocab_size,
        embedding_dim=256,
        num_layers=6,
        num_heads=8,
        ffn_dim=1024,
        max_seq_len=max_seq_len + 1,
        dropout=0.1,
    ).to(device)

    std_result = train_model(
        std_model, train_data, val_data, tok, vocab_size, device,
        "Standard Transformer", num_epochs=num_epochs, batch_size=batch_size,
        grad_accum=grad_accum, peak_lr=peak_lr, use_amp=use_amp,
        autocast_device=autocast_device, autocast_dtype=autocast_dtype,
        save_dir="bpe_std_checkpoints",
    )
    results.append(std_result)

    print(f"\n  --- Standard Transformer Generation ---")
    for seed in ["The president of the", "In the year", "Scientists discovered that",
                 "The city of New York", "He was born in"]:
        text = generate_text(std_model, tok, seed, device, max_tokens=40)
        print(f"  [{seed}] -> {text}")

    test_loader = create_dataloader(test_data, batch_size=batch_size, shuffle=False)
    std_tl, std_tp, std_ta = evaluate(
        std_model,
        test_loader,
        vocab_size,
        device,
        use_amp=use_amp,
        autocast_device=autocast_device,
        autocast_dtype=autocast_dtype,
    )
    std_result['test_ppl'] = std_tp
    std_result['test_acc'] = std_ta
    print(f"\n  TEST: PPL {std_tp:.1f} | Acc {std_ta:.1f}%")

    del std_model
    torch.cuda.empty_cache()

    # ============================================================
    # MODEL 2: DCWT-v2
    # ============================================================
    print(f"\n{'='*65}")
    print(f"  MODEL 2: DCWT-v2 ({dcwt_tree_mode})")
    print(f"{'='*65}")

    wave_model = DCWTv2Transformer(
        vocab_size=vocab_size,
        embedding_dim=256,
        num_layers=6,
        num_heads=8,
        ffn_dim=1024,
        max_seq_len=max_seq_len + 1,
        k_max=8,
        local_window=32,
        tree_mode=dcwt_tree_mode,
        dropout=0.1,
        use_checkpoint=False,
    ).to(device)

    wave_result = train_model(
        wave_model, train_data, val_data, tok, vocab_size, device,
        "DCWT-v2", num_epochs=num_epochs, batch_size=batch_size,
        grad_accum=grad_accum, peak_lr=peak_lr, use_amp=use_amp,
        autocast_device=autocast_device, autocast_dtype=autocast_dtype,
        save_dir="bpe_dcwt_v2_checkpoints",
    )
    results.append(wave_result)

    print(f"\n  --- DCWT-v2 Generation ---")
    for seed in ["The president of the", "In the year", "Scientists discovered that",
                 "The city of New York", "He was born in"]:
        text = generate_text(wave_model, tok, seed, device, max_tokens=40)
        print(f"  [{seed}] -> {text}")

    test_loader = create_dataloader(test_data, batch_size=batch_size, shuffle=False)
    wave_tl, wave_tp, wave_ta = evaluate(
        wave_model,
        test_loader,
        vocab_size,
        device,
        use_amp=use_amp,
        autocast_device=autocast_device,
        autocast_dtype=autocast_dtype,
    )
    wave_result['test_ppl'] = wave_tp
    wave_result['test_acc'] = wave_ta
    print(f"\n  TEST: PPL {wave_tp:.1f} | Acc {wave_ta:.1f}%")

    del wave_model
    torch.cuda.empty_cache()

    # ============================================================
    # COMPARISON
    # ============================================================
    print(f"\n{'='*65}")
    print("  BPE BENCHMARK RESULTS")
    print(f"{'='*65}")

    print(f"\n  {'Metric':<20} {'Std Transformer':>18} {'DCWT-v2':>18} {'Winner':>10}")
    print(f"  {'-'*20} {'-'*18} {'-'*18} {'-'*10}")

    s_tp, w_tp = results[0]['test_ppl'], results[1]['test_ppl']
    s_ta, w_ta = results[0]['test_acc'], results[1]['test_acc']

    winner_p = "DCWT-v2" if w_tp < s_tp else "Standard"
    winner_a = "DCWT-v2" if w_ta > s_ta else "Standard"
    print(f"  {'Test PPL':<20} {s_tp:>18.1f} {w_tp:>18.1f} {winner_p:>10}")
    print(f"  {'Test Accuracy':<20} {s_ta:>17.1f}% {w_ta:>17.1f}% {winner_a:>10}")
    print(f"  {'Parameters':<20} {results[0]['params']:>18,} {results[1]['params']:>18,}")
    print(f"  {'Train Time':<20} {results[0]['total_time']/60:>17.1f}m {results[1]['total_time']/60:>17.1f}m")
    print(f"  {'Complexity':<20} {'O(n^2)':>18} {'O(n log n)':>18}")
    print(f"  {'Tokenizer':<20} {'BPE '+str(vocab_size):>18} {'BPE '+str(vocab_size):>18}")
    print(f"  {'Seq Length':<20} {max_seq_len:>18} {max_seq_len:>18}")

    if w_tp < s_tp:
        pct = (s_tp - w_tp) / s_tp * 100
        print(f"\n  DCWT-v2 BEATS Standard Transformer by {pct:.1f}% on test PPL!")
    else:
        pct = (w_tp - s_tp) / s_tp * 100
        print(f"\n  Standard beats DCWT-v2 by {pct:.1f}% — but DCWT-v2 uses O(n log n)")

    with open("bpe_dcwt_v2_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*65}")
    print("  BENCHMARK COMPLETE")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
