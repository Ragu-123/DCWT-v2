# Wave Field LLM: HALE-First Architecture

This repository now uses **HALE** (Hierarchical Adaptive Linear-Enhanced Attention) as the default accelerated architecture, while keeping **DCWT-v2** available for compatibility and ablation.

Reference research files:
- `newPOC/HALE_RESEARCH.md`
- `newPOC/hale_attention.py`
- `newPOC/benchmark_hale.py`

## What Is Default Now

- **Primary model path**: `src/hale_attention.py` (`HALETransformer`, `HALEAttention`)
- **Compatibility aliases**:
  - `src/wave_field_transformer.py` -> `WaveFieldTransformer` aliases HALE
  - `src/wave_field_attention.py` -> `WaveFieldAttention` aliases HALE
- **Legacy model path**: `src/dcwt_v2.py` (still available)

## HALE Design Summary

HALE combines:
1. Local causal SDPA window (exact short-range precision)
2. Causal linear attention (ELU+1 feature map) for global memory
3. Causal Haar multi-scale context
4. Adaptive local/global gating
5. Incremental generation cache (`O(d^2)`-state style updates per layer)

## Installation

```bash
pip install -r requirements.txt
```

PowerShell:

```powershell
$env:PYTHONPATH='.;tokenizers'
```

## Benchmark Entry Points

### WikiText-2 comparison (Standard vs accelerated model)

```powershell
$env:WFL_ACCEL_MODEL='hale'   # default; options: hale | dcwt
python benchmarks/benchmark_wikitext2.py
```

### BPE benchmark

```powershell
$env:WFL_ACCEL_MODEL='hale'   # default; options: hale | dcwt
python benchmarks/train_wave_v35_bpe.py
```

### 100M-scale benchmark

```powershell
$env:WFL_ACCEL_MODEL='hale'   # default; options: hale | dcwt
python benchmarks/train_100m_bpe.py
```

### Dedicated HALE benchmark

```powershell
python benchmarks/benchmark_hale.py
```

## Configuration Knobs

### HALE knobs

```powershell
$env:HALE_LOCAL_WINDOW='64'
$env:HALE_NUM_HAAR_LEVELS='4'
$env:HALE_CHUNK_SIZE='128'
```

### Optional DCWT bio-flow knobs (used only when `WFL_ACCEL_MODEL=dcwt`)

```powershell
$env:DCWT_BIO_FLOW='1'
$env:DCWT_TREE_MODE='full'
$env:DCWT_JACOBI_ITERS='2'
$env:DCWT_DEPTH_COND_GWM='1'
$env:DCWT_HAAR_INIT='1'
$env:DCWT_HEARTBEAT='1'
$env:DCWT_SLIME_MOLD='1'
```

## Project Structure

```text
src/
  hale_attention.py           # HALE architecture (default)
  dcwt_v2.py                  # DCWT-v2 architecture (legacy/ablation)
  wave_field_transformer.py   # Compatibility alias to HALE
  wave_field_attention.py     # Compatibility alias to HALE
benchmarks/
  benchmark_wikitext2.py      # Standard vs accelerated (HALE default)
  train_wave_v35_bpe.py       # BPE benchmark (HALE default)
  train_100m_bpe.py           # 100M benchmark (HALE default)
  benchmark_hale.py           # Dedicated HALE comparison script
  wikitext2.py                # Convenience entrypoint
diagnostics/
  diagnose_bpe.py
tests/
  test_causality.py
  test_dcwt_v2.py
```

## Compatibility Notes

- Existing imports like `from src.wave_field_transformer import WaveFieldTransformer` still work.
- Existing DCWT scripts remain usable by setting `WFL_ACCEL_MODEL=dcwt`.

## License

AGPL-3.0 (see `LICENSE`).
