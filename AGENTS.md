# Repository Guidelines

## Project Structure & Module Organization
- `src/`: core model implementations with `dcwt_v2.py` as the primary architecture; `wave_field_*` files are compatibility aliases.
- `benchmarks/`: training and comparison scripts (`wikitext2.py`, `benchmark_wikitext2.py`, `train_dcwt_v2_bpe.py`, `train_dcwt_v2_100m.py`).
- `diagnostics/`: DCWT-v2 analysis scripts for structural and BPE diagnostics.
- `tokenizers/`: tokenizer implementations (`field_tokenizer_v2.py`, `field_tokenizer_v3.py`, `field_aware_tokenizer.py`).
- `tests/`: regression checks (currently `test_causality.py`).
- `docs/`: benchmark results and supporting technical notes.

## Build, Test, and Development Commands
```bash
pip install -r requirements.txt
```
Install runtime dependencies (`torch`, `datasets`, `tokenizers`, etc.).

```powershell
$env:PYTHONPATH='.;tokenizers'
python tests/test_causality.py
python tests/test_dcwt_v2.py
```
Run the core DCWT-v2 regression tests.

```powershell
$env:PYTHONPATH='.;tokenizers'
python benchmarks/train_wave_v35_bpe.py
python benchmarks/train_dcwt_v2_bpe.py
python benchmarks/wikitext2.py
python benchmarks/benchmark_wikitext2.py
python diagnostics/diagnose_bpe.py
```
Run training/benchmark/diagnostic workflows. These scripts write checkpoints and JSON results in the repo root (for example `bpe_dcwt_v2_checkpoints/` and `*_dcwt_v2_results.json`).

## Coding Style & Naming Conventions
- Use Python with 4-space indentation and PEP 8 conventions.
- Naming: `snake_case` for functions/variables/files, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Keep tensor-shape handling explicit (`B, N, D` style) and preserve causality assumptions in attention code paths.
- No formatter/linter config is committed; match existing style and keep diffs focused.

## Testing Guidelines
- Add tests in `tests/` for behavior changes, especially in attention, scattering, or FFT logic.
- Test files should be named `test_<feature>.py`; test functions should be `test_<scenario>()`.
- Prefer deterministic checks with tolerances, and include at least one future-token leakage assertion when modifying causal logic.

## Commit & Pull Request Guidelines
- Follow concise, imperative commit subjects (for example: `Delete ...`, `Change ...`, `Add ...`).
- Keep commits scoped to one logical change.
- PRs should include a brief summary, rationale, exact reproduction commands, and before/after metrics for model-quality or performance changes.
