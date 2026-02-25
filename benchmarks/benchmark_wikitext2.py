"""
WikiText-2 Benchmark: Standard Transformer vs DCWT-v2
=====================================================
Side-by-side benchmark with matched data and optimizer settings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import math
import json
from typing import Any, Dict, Optional

from src.dcwt_v2 import DCWTv2Transformer
from src.bio_scheduler import SlimeMoldKScheduler, heartbeat_schedule
from field_tokenizer_v2 import FieldTokenizerV2


# ======================================================================
# STANDARD TRANSFORMER BASELINE (PyTorch native, O(n^2) attention)
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
# SHARED UTILITIES
# ======================================================================

def load_wikitext2():
    """Load WikiText-2 using HuggingFace datasets."""
    from datasets import load_dataset
    print("Loading WikiText-2 from HuggingFace...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")

    splits = {}
    for split_name, hf_split in [('train', 'train'), ('valid', 'validation'), ('test', 'test')]:
        lines = []
        for item in ds[hf_split]:
            text = item['text'].strip()
            if text and not text.startswith('='):
                lines.append(text)
        splits[split_name] = lines

    print("WikiText-2 loaded.")
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


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _is_dcwt_model(model: nn.Module) -> bool:
    return isinstance(model, DCWTv2Transformer)


def _model_forward(
    model: nn.Module,
    x: torch.Tensor,
    bio_cfg: Optional[Dict[str, Any]] = None,
):
    bio_cfg = bio_cfg or {}
    if _is_dcwt_model(model):
        return model(x, jacobi_iters=int(bio_cfg.get("jacobi_iters", 0)))
    return model(x)


def create_batches(data, batch_size, device, shuffle=True):
    if shuffle:
        indices = torch.randperm(len(data)).tolist()
    else:
        indices = list(range(len(data)))
    batches = []
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        bx = [data[i][0] for i in batch_idx]
        by = [data[i][1] for i in batch_idx]
        ml = max(x.size(0) for x in bx)
        px = torch.zeros(len(bx), ml, dtype=torch.long, device=device)
        py = torch.full((len(by), ml), -100, dtype=torch.long, device=device)
        for i, (x, y) in enumerate(zip(bx, by)):
            px[i, :x.size(0)] = x
            py[i, :y.size(0)] = y
        batches.append((px, py))
    return batches


@torch.no_grad()
def evaluate(
    model,
    batches,
    vocab_size,
    device,
    use_amp=False,
    bio_cfg: Optional[Dict[str, Any]] = None,
):
    model.eval()
    tl, tc, tt, n = 0, 0, 0, 0
    for x, y in batches:
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits, _ = _model_forward(model, x, bio_cfg)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1), ignore_index=-100)
        tl += loss.item(); n += 1
        mask = y != -100
        tc += (logits.argmax(-1)[mask] == y[mask]).sum().item()
        tt += mask.sum().item()
    model.train()
    al = tl / max(n, 1)
    return al, math.exp(min(al, 20)), tc / max(tt, 1) * 100


@torch.no_grad()
def generate_text(model, tokenizer, seed, device, max_tokens=50,
                  temperature=0.7, top_k=40, top_p=0.9, rep_penalty=1.5,
                  bio_cfg: Optional[Dict[str, Any]] = None):
    model.eval()
    ids = tokenizer.encode(seed.lower())
    if not ids:
        return seed + " [empty]"
    bio_cfg = bio_cfg or {}
    if _is_dcwt_model(model):
        inp = torch.tensor(ids, device=device).unsqueeze(0)
        out = model.generate(
            inp,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=rep_penalty,
            jacobi_iters=int(bio_cfg.get("jacobi_iters", 0)),
        )
        model.train()
        return tokenizer.decode(out[0].tolist())

    gen = ids.copy()
    inp = torch.tensor(ids, device=device).unsqueeze(0)
    for _ in range(max_tokens):
        logits, _ = model(inp)
        nl = logits[0, -1, :] / temperature
        for pid in set(gen[-20:]):
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
    return tokenizer.decode(gen)


def encode_lines(lines, tokenizer, max_seq_len):
    """Convert text lines to (input, target) pairs."""
    data = []
    for line in lines:
        ids = tokenizer.encode(line)
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


def train_model(model, train_data, val_data, vocab_size, device,
                model_name, num_epochs=30, batch_size=32, peak_lr=0.0003,
                use_amp=True, save_dir="checkpoints",
                bio_cfg: Optional[Dict[str, Any]] = None):
    """Train a model and return results."""
    os.makedirs(save_dir, exist_ok=True)
    bio_cfg = bio_cfg or {}
    use_bio = _is_dcwt_model(model) and bool(bio_cfg.get("enabled", False))

    params = sum(p.numel() for p in model.parameters())
    print(f"\n  {model_name}: {params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01, eps=1e-8)
    spe = math.ceil(len(train_data) / batch_size)
    scheduler = WarmupCosineScheduler(optimizer, spe * 3, spe * num_epochs, min_lr=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    slime = None
    if use_bio and bio_cfg.get("use_slime_mold", False):
        slime = SlimeMoldKScheduler(
            model,
            alpha=float(bio_cfg.get("slime_alpha", 0.3)),
            mu=float(bio_cfg.get("slime_mu", 0.3)),
            update_every=int(bio_cfg.get("slime_update_every", 500)),
        )
    slime_start_step = int(bio_cfg.get("slime_start_step", 8000))
    global_step = 0

    best_vl, best_vp, best_va, best_ep = float('inf'), float('inf'), 0, 0

    t0 = time.time()
    for epoch in range(1, num_epochs + 1):
        et = time.time()
        model.train()
        batches = create_batches(train_data, batch_size, device)
        tl, nb = 0, 0
        for x, y in batches:
            if use_bio and bio_cfg.get("use_heartbeat", False):
                heartbeat_schedule(global_step, model)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits, _ = _model_forward(model, x, bio_cfg)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1), ignore_index=-100)
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if slime is not None and global_step >= slime_start_step:
                slime.accumulate(model.get_tree_grad_norms())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            global_step += 1
            if slime is not None and global_step >= slime_start_step:
                slime.update()
            tl += loss.item(); nb += 1
        al = tl / max(nb, 1)
        et = time.time() - et

        if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
            vb = create_batches(val_data, batch_size, device, shuffle=False)
            vl, vp, va = evaluate(model, vb, vocab_size, device, use_amp, bio_cfg=bio_cfg)
            if vl < best_vl:
                best_vl, best_vp, best_va, best_ep = vl, vp, va, epoch
                torch.save(model.state_dict(), os.path.join(save_dir, "best.pt"))
                mk = " * BEST"
            else:
                mk = ""
            print(f"  Ep {epoch:3d}/{num_epochs} | Train {al:.4f} | Val {vl:.4f} PPL {vp:.1f} Acc {va:.1f}% | {et:.1f}s{mk}")
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
# MAIN BENCHMARK
# ======================================================================

def main():
    bio_enabled = _env_flag("DCWT_BIO_FLOW", True)
    dcwt_tree_mode = os.environ.get("DCWT_TREE_MODE", "full" if bio_enabled else "flash_only")
    if bio_enabled and dcwt_tree_mode != "full":
        print("Bio flow requires tree_mode='full'; overriding requested tree_mode.")
        dcwt_tree_mode = "full"
    bio_cfg: Dict[str, Any] = {
        "enabled": bio_enabled,
        "jacobi_iters": int(os.environ.get("DCWT_JACOBI_ITERS", "2" if bio_enabled else "0")),
        "use_depth_conditioned_gwm": _env_flag("DCWT_DEPTH_COND_GWM", bio_enabled),
        "depth_embed_dim": int(os.environ.get("DCWT_DEPTH_EMBED_DIM", "32")),
        "use_haar_init": _env_flag("DCWT_HAAR_INIT", bio_enabled),
        "use_heartbeat": _env_flag("DCWT_HEARTBEAT", bio_enabled),
        "use_slime_mold": _env_flag("DCWT_SLIME_MOLD", bio_enabled),
        "slime_alpha": float(os.environ.get("DCWT_SLIME_ALPHA", "0.3")),
        "slime_mu": float(os.environ.get("DCWT_SLIME_MU", "0.3")),
        "slime_update_every": int(os.environ.get("DCWT_SLIME_UPDATE_EVERY", "500")),
        "slime_start_step": int(os.environ.get("DCWT_SLIME_START_STEP", "8000")),
        "compile_gwm": _env_flag("DCWT_COMPILE_GWM", True),
    }
    if not bio_enabled:
        bio_cfg["jacobi_iters"] = 0

    print("=" * 65)
    print("  WIKITEXT-2 BENCHMARK")
    print("  Standard Transformer (O(n^2)) vs DCWT-v2 (mode-selectable)")
    print("  Same data, tokenizer, training setup — fair comparison")
    print(f"  DCWT tree_mode: {dcwt_tree_mode}")
    print(
        "  Bio flow:"
        f" enabled={bio_cfg['enabled']}"
        f" depth_cond_gwm={bio_cfg['use_depth_conditioned_gwm']}"
        f" haar={bio_cfg['use_haar_init']}"
        f" jacobi_iters={bio_cfg['jacobi_iters']}"
        f" heartbeat={bio_cfg['use_heartbeat']}"
        f" slime={bio_cfg['use_slime_mold']}"
    )
    print("=" * 65)

    # Load WikiText-2
    splits = load_wikitext2()

    total_train_lines = len(splits['train'])
    total_val_lines = len(splits['valid'])
    total_test_lines = len(splits['test'])
    print(f"\nWikiText-2 loaded:")
    print(f"  Train: {total_train_lines:,} lines")
    print(f"  Valid: {total_val_lines:,} lines")
    print(f"  Test:  {total_test_lines:,} lines")

    # Build shared tokenizer on training data
    tok = FieldTokenizerV2(field_size=1024)
    tok.build_vocab(splits['train'])
    vocab_size = tok.vocab_size_actual()

    # Encode data
    max_seq_len = 128
    train_data = encode_lines(splits['train'], tok, max_seq_len)
    val_data = encode_lines(splits['valid'], tok, max_seq_len)
    test_data = encode_lines(splits['test'], tok, max_seq_len)

    print("\nTokenizer: FieldTokenizerV2 (field_size=1024)")
    print(f"Vocab: {vocab_size} | max_seq_len: {max_seq_len}")
    print(f"Train sequences: {len(train_data):,}")
    print(f"Val sequences:   {len(val_data):,}")
    print(f"Test sequences:  {len(test_data):,}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f"\nDevice: {device} | AMP: {use_amp}")
    if device.type == "cuda" and hasattr(torch.backends, "cuda"):
        try:
            print(
                "SDPA backends:"
                f" flash={torch.backends.cuda.flash_sdp_enabled()}"
                f" mem_efficient={torch.backends.cuda.mem_efficient_sdp_enabled()}"
                f" math={torch.backends.cuda.math_sdp_enabled()}"
            )
        except Exception:
            pass

    num_epochs = 30
    batch_size = 32
    peak_lr = 0.0003
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
        std_model, train_data, val_data, vocab_size, device,
        "Standard Transformer", num_epochs=num_epochs, batch_size=batch_size,
        peak_lr=peak_lr, use_amp=use_amp, save_dir="wikitext2_std_checkpoints",
        bio_cfg={"enabled": False},
    )
    results.append(std_result)

    # Generate samples
    print(f"\n  --- Standard Transformer Generation ---")
    for seed in ["The president of", "In the year", "Scientists discovered that"]:
        text = generate_text(std_model, tok, seed, device, max_tokens=40, bio_cfg={"enabled": False})
        print(f"  [{seed}] -> {text}")

    # Final test eval
    test_batches = create_batches(test_data, batch_size, device, shuffle=False)
    std_test_loss, std_test_ppl, std_test_acc = evaluate(
        std_model,
        test_batches,
        vocab_size,
        device,
        use_amp,
        bio_cfg={"enabled": False},
    )
    std_result['test_ppl'] = std_test_ppl
    std_result['test_acc'] = std_test_acc
    print(f"\n  TEST SET: PPL {std_test_ppl:.1f} | Acc {std_test_acc:.1f}%")

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
        use_depth_conditioned_gwm=bool(bio_cfg["use_depth_conditioned_gwm"]),
        depth_embed_dim=int(bio_cfg["depth_embed_dim"]),
        compile_gwm=bool(bio_cfg["compile_gwm"]),
        use_haar_init=bool(bio_cfg["use_haar_init"]),
    ).to(device)

    wave_result = train_model(
        wave_model, train_data, val_data, vocab_size, device,
        "DCWT-v2", num_epochs=num_epochs, batch_size=batch_size,
        peak_lr=peak_lr, use_amp=use_amp, save_dir="wikitext2_dcwt_v2_checkpoints",
        bio_cfg=bio_cfg,
    )
    results.append(wave_result)

    # Generate samples
    print(f"\n  --- DCWT-v2 Generation ---")
    for seed in ["The president of", "In the year", "Scientists discovered that"]:
        text = generate_text(wave_model, tok, seed, device, max_tokens=40, bio_cfg=bio_cfg)
        print(f"  [{seed}] -> {text}")

    # Final test eval
    test_batches = create_batches(test_data, batch_size, device, shuffle=False)
    wave_test_loss, wave_test_ppl, wave_test_acc = evaluate(
        wave_model,
        test_batches,
        vocab_size,
        device,
        use_amp,
        bio_cfg=bio_cfg,
    )
    wave_result['test_ppl'] = wave_test_ppl
    wave_result['test_acc'] = wave_test_acc
    print(f"\n  TEST SET: PPL {wave_test_ppl:.1f} | Acc {wave_test_acc:.1f}%")

    del wave_model
    torch.cuda.empty_cache()

    # ============================================================
    # FINAL COMPARISON
    # ============================================================
    print(f"\n{'='*65}")
    print("  WIKITEXT-2 BENCHMARK RESULTS")
    print(f"{'='*65}")

    print(f"\n  {'Metric':<20} {'Std Transformer':>18} {'DCWT-v2':>18} {'Winner':>10}")
    print(f"  {'-'*20} {'-'*18} {'-'*18} {'-'*10}")

    # Val PPL
    s_vp = results[0]['best_ppl']
    w_vp = results[1]['best_ppl']
    winner_vp = "DCWT-v2" if w_vp < s_vp else "Standard"
    print(f"  {'Val PPL':<20} {s_vp:>18.1f} {w_vp:>18.1f} {winner_vp:>10}")

    # Val Acc
    s_va = results[0]['best_acc']
    w_va = results[1]['best_acc']
    winner_va = "DCWT-v2" if w_va > s_va else "Standard"
    print(f"  {'Val Accuracy':<20} {s_va:>17.1f}% {w_va:>17.1f}% {winner_va:>10}")

    # Test PPL
    s_tp = results[0]['test_ppl']
    w_tp = results[1]['test_ppl']
    winner_tp = "DCWT-v2" if w_tp < s_tp else "Standard"
    print(f"  {'Test PPL':<20} {s_tp:>18.1f} {w_tp:>18.1f} {winner_tp:>10}")

    # Test Acc
    s_ta = results[0]['test_acc']
    w_ta = results[1]['test_acc']
    winner_ta = "DCWT-v2" if w_ta > s_ta else "Standard"
    print(f"  {'Test Accuracy':<20} {s_ta:>17.1f}% {w_ta:>17.1f}% {winner_ta:>10}")

    # Params
    print(f"  {'Parameters':<20} {results[0]['params']:>18,} {results[1]['params']:>18,}")

    # Time
    s_t = results[0]['total_time']
    w_t = results[1]['total_time']
    print(f"  {'Train Time':<20} {s_t/60:>17.1f}m {w_t/60:>17.1f}m")

    # Complexity
    print(f"  {'Complexity':<20} {'O(n^2)':>18} {'O(n log n)':>18}")

    # PPL improvement
    if w_tp < s_tp:
        pct = (s_tp - w_tp) / s_tp * 100
        print(f"\n  DCWT-v2 beats Standard Transformer by {pct:.1f}% on WikiText-2 test PPL")
        print(f"  while using O(n log n) vs O(n^2) complexity.")
    else:
        pct = (w_tp - s_tp) / s_tp * 100
        print(f"\n  Standard Transformer beats DCWT-v2 by {pct:.1f}% on WikiText-2 test PPL")
        print(f"  but DCWT-v2 uses O(n log n) vs O(n^2) complexity.")

    # Save results
    with open("wikitext2_dcwt_v2_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*65}")
    print("  BENCHMARK COMPLETE")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
