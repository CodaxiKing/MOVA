# Architecture Audit — before the runtime refactor (2026-09-24)

Scope: Phase 0 of the architecture evolution (runtime / device / model interface). Audited at commit
`36b5a08` (branch `refactor/runtime-architecture`, forked from `main@a21409c`). Tests before any change:
**64 passed in 27.79s** (CPU). Evidence: `outputs/arch-audit/env_before.json`, `benchmark/baseline/`.

## Environment where this was audited

| Item | Value |
|---|---|
| Machine | HP ProBook 640 G8, Intel Iris Xe, **no CUDA** |
| Python / PyTorch | 3.12.10 / 2.14.0+cpu (CUDA build: None) |
| RAM | 15.69 GB total, **0.84 GB available** during the session |
| Wan weights | not cached (0/17 files, 19.04 GB missing, download not authorised) |
| User media | none (`assets/` empty) |

## Current architecture (as found)

Flat top-level packages (not `mova/…`):

```text
scripts/inference_baseline.py   ← all orchestration (args, config, hardware, cache, preflight, control, run record, output)
  ├─ common/env.py              detect_hardware (torch.cuda, index 0 only) + select_profile (device/dtype/offload/res/frames)
  ├─ common/hf_utils.py         manifest + cache verification (ADR-008)
  ├─ common/resources.py        disk/RAM/VRAM preflight (torch.cuda index 0)
  ├─ preprocessing/pipeline.py  MediaPipe extraction → pose_openpose.mp4
  ├─ inference/conditioning.py  letterbox, 4k+1, resolution
  ├─ inference/baseline_vace.py UMT5 prompt cache, load_pipeline (pipe.to("cuda")), generate (torch.cuda stats)
  └─ evaluation/video.py        output integrity
scripts/benchmark.py            → subprocess scripts/inference_baseline.py per case
models/, training/              empty packages (only __init__.py)
```

Inference flow: `reference.png + motion.mp4 → MediaPipe → pose_openpose.mp4 → letterbox/4k+1 →
UMT5 (CPU, cached) → WanVACEPipeline (offload, VAE tiling) → frames → integrity check → output.mp4 + run.json`.
Training flow: **does not exist** (`training/` empty).

Dependencies: torch, diffusers 0.40, transformers 5.x, mediapipe, opencv, imageio-ffmpeg, pyyaml, psutil,
huggingface_hub. Not installed: fastapi/uvicorn, onnx, onnxruntime, tensorrt. The package is not
installed in the venv (no `mova` console command).

Model: Wan2.1-VACE-1.3B-diffusers @ `ec4d2cb0…` (ADR-001/002/008). Never run with real weights.

## Coupling points

| Where | Coupling | Risk |
|---|---|---|
| `inference/baseline_vace.py::load_pipeline` | `pipe.to("cuda")` hard-coded; offload always to the default CUDA device | cannot choose GPU N, CPU or ROCm device string |
| `inference/baseline_vace.py::generate` | `torch.cuda.reset_peak_memory_stats / max_memory_*` inside the model code | memory accounting mixed with model code |
| `common/env.py::detect_hardware` | `torch.cuda.*` on device 0 only; `device = "cuda" if … else "cpu"` | multi-GPU and ROCm invisible |
| `common/env.py::select_profile` | device, dtype, offload (hardware decisions) mixed with resolution/frames (model defaults) | precision/offload policy not reusable |
| `common/resources.py::vram_free_gb` | `torch.cuda.mem_get_info(0)` | wrong GPU checked on multi-GPU |
| `scripts/inference_baseline.py` | the only entry point; model-specific (manifest, UMT5) and generic steps interleaved | a CLI/API would have to duplicate it |
| `configs/baseline.yaml` | `model.dtype`, `model.offload` are runtime concerns inside the model section | runtime not configurable separately |
| `scripts/check_env.py` | CUDA smoke test inline | fine, diagnostic only |

Existing abstractions to reuse (not duplicate): `common.config` (YAML + `--set` overrides),
`common.experiment.ExperimentRun`, `common.hf_utils` manifest/cache, `common.resources` preflight,
`evaluation.video` integrity, `inference.conditioning`. There was **no** runtime, device, model interface,
registry, capability check or structured error type.

## Migration plan

1. `runtime/` — `Runtime` interface, `PyTorchRuntime`, `RuntimeManager`, `DeviceManager`,
   `PrecisionManager`, `MemoryManager`. Only place allowed to call `torch.cuda.*` (plus the diagnostic script).
2. `common/errors.py` — structured errors shared by all layers.
3. `models/base.py` + `models/registry.py` + `models/backbones/wan_vace.py` — the current VACE baseline behind
   a model interface; `inference/baseline_vace.py` keeps the numerics (no placement, no CUDA calls).
4. `core/` — one inference service used by the script, the CLI and (later) the API.
5. `configs/runtime.yaml` — runtime/device/precision/offload defaults.
6. `mova/cli.py` — `mova info`, `mova infer`, … ; `scripts/inference_baseline.py` becomes a thin wrapper.

Layering (no cycles): `common ← runtime ← models ← core ← {mova CLI, scripts, future API}`.

## Risks

- Changing numerics of the generation path → mitigated by the bit-for-bit tiny-pipeline baseline.
- Breaking `scripts/benchmark.py` (it runs `inference_baseline.py` by subprocess and reads `settings`,
  `stats.generation_time_s`, `stats.vram_peak_gb`, `hardware`, `git_commit` from run.json) → keep those fields.
- Resuming a benchmark across the refactor is refused by design (generation code hash changes) — expected.
- CUDA placement/offload code paths cannot be executed on this machine → stay UNVERIFIED until the RTX 3060.

## Baseline recorded

`benchmark/baseline/tiny_vace_cpu.json` (script `benchmark/baseline/capture_tiny_vace.py`): real Diffusers
WanVACE code path, tiny random weights, 32×32, 5 frames, 2 steps, fp32, seed 42, CPU.
Deterministic across 3 repeats, output SHA-256 `a3cbab52523b48c2…`, finite. This pins numerics only; it says
nothing about quality, VRAM or speed of the real 1.3B model. The real baseline (EXP-001) is **BLOCKED**.

## Tests needed

Runtime / device (incl. invalid and missing devices) / precision / memory / capability validation /
registry / config precedence / CLI (valid, invalid, missing file, unknown model, unknown runtime) /
end-to-end inference through the CLI with the tiny model / bit-for-bit regression against the baseline.
