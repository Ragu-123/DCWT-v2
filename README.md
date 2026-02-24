# DCWT-v2: Gated Multi-Vector Dyadic Causal Wave Tree

DCWT-v2 is the updated architecture for this repository. It replaces the V3.x field-convolution path with a causal tree attention system designed to address information-capacity collapse while preserving `O(n log n)` training complexity.

The full design rationale is captured in `newproposal.txt`.

## What Changed

The codebase now implements the five proposal fixes end-to-end:

1. **Gated Wave Merge (GWM)**: content-adaptive gating during parent-node merge.
2. **Multi-Vector Node Bank (MVNB)**: each internal node stores up to `K_max` vectors.
3. **Complementary Dual Coverage (CDC)**: exact local window + compressed long-range tree retrieval.
4. **Depth-Decomposed Queries (DDQ)**: depth-specific query projections and temperatures.
5. **Tree LayerNorm + Skip**: normalization and stable skip paths in deep merges.

## Complexity

- Training: `O(n log n)` per layer (with constant-factor `K_max` overhead).
- Inference target: `O(log n)` per token with tree cache + `O(1)` local window.

## Quick Start

```bash
pip install -r requirements.txt
```

PowerShell examples:

```powershell
$env:PYTHONPATH='.;tokenizers'
python benchmarks/wikitext2.py
python benchmarks/train_dcwt_v2_bpe.py
python benchmarks/train_dcwt_v2_100m.py
python diagnostics/diagnose_physics.py
python diagnostics/diagnose_bpe.py
python tests/test_causality.py
python tests/test_dcwt_v2.py
```

## Project Structure

```text
src/
  dcwt_v2.py                 # Core DCWT-v2 implementation
  wave_field_transformer.py  # Backward-compatible alias to DCWT-v2
  wave_field_attention.py    # Backward-compatible alias to DCWT-v2 attention
benchmarks/
  wikitext2.py               # WikiText-2 benchmark entrypoint
  benchmark_wikitext2.py     # Standard vs DCWT-v2 benchmark
  train_wave_v35_bpe.py      # BPE benchmark (kept filename, now DCWT-v2)
  train_dcwt_v2_bpe.py       # Alias entrypoint for DCWT-v2 BPE benchmark
  train_100m_bpe.py          # 100M-scale BPE benchmark (DCWT-v2)
  train_dcwt_v2_100m.py      # Alias entrypoint for DCWT-v2 100M benchmark
diagnostics/
  diagnose_physics.py        # DCWT-v2 structural diagnostics
  diagnose_bpe.py            # DCWT-v2 BPE diagnostics
tests/
  test_causality.py          # Future-token leak regression
  test_dcwt_v2.py            # Structural smoke tests
```

## Compatibility Notes

- Existing imports such as `from src.wave_field_transformer import WaveFieldTransformer` still work.
- Legacy constructor args (`field_size`, `interference_interval`, `device`) are accepted for compatibility, but DCWT-v2 uses `k_max` and `local_window` for its core behavior.

## License

This project is licensed under AGPL-3.0. See `LICENSE`.
