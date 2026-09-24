# Regression: architecture refactor (ADR-009)

| File | What |
|---|---|
| `../baseline/capture_tiny_vace.py` → `../baseline/tiny_vace_cpu.json` | Phase 0, **before** the refactor: pre-refactor `generate()` with a tiny random WanVACE (32×32, 5 frames, 2 steps, fp32, seed 42, CPU), 3 repeats |
| `tiny_vace_regression.py` → `tiny_vace_cpu.json` | same inputs through `registry → WanVACETinyRandomModel → PyTorchRuntime`; fp32 must be bit-identical; bf16/fp16 must be finite |

```bash
.venv/Scripts/python benchmark/regression/tiny_vace_regression.py   # exit 0 = bit-exact + finite
```

Result 2026-09-24 (HP ProBook, i7-1165G7, CPU only, torch 2.14.0+cpu):

| Metric | Baseline | New architecture |
|---|---|---|
| fp32 output SHA-256 | `a3cbab52523b48c2…` | identical (3/3 runs) |
| generation (warm) | ~0.09 s | ~0.09 s |
| warm total per run | 0.12 s | 0.26–0.31 s |

The total difference is `unload()` → `MemoryManager.cleanup()` → `gc.collect()` (~0.14 s, measured per stage):
a fixed cost per model unload, intentional, negligible against a real generation. Runtime/config/configure
overhead < 25 ms. Peak VRAM/RAM, startup and throughput of the **real** 1.3B model are **not measured**
(no CUDA, no weights): the real regression (EXP-001 before vs after) is BLOCKED.
