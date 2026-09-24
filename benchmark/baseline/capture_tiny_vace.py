"""Phase 0 baseline of the generation path, reproducible on any machine without weights.

Runs the REAL Diffusers WanVACEPipeline code through `inference.baseline_vace.generate` with a tiny,
randomly initialised model (seeded) on CPU, and records the output hash, shape, finiteness and timings.
This is NOT a quality baseline (output is noise): it pins the numerical behaviour of the code path so the
architecture refactor can be checked bit-for-bit. The real Wan2.1-VACE-1.3B baseline is EXP-001 (blocked:
no CUDA, weights not downloaded).

This file is frozen on purpose: it builds the tiny pipeline inline and calls the pre-refactor API.

Usage: python benchmark/baseline/capture_tiny_vace.py [--out benchmark/baseline/tiny_vace_cpu.json] [--repeat 3]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

# Inputs and model shape shared with benchmark/regression (must stay identical).
H, W, FRAMES, STEPS, SEED = 32, 32, 5, 2, 42


def tiny_vace_pipe():
    from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanVACEPipeline, WanVACETransformer3DModel

    torch.manual_seed(0)
    vae = AutoencoderKLWan(base_dim=3, z_dim=16, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
                           temperal_downsample=[False, True, True])
    transformer = WanVACETransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12, in_channels=16, out_channels=16,
        text_dim=32, freq_dim=32, ffn_dim=32, num_layers=3, cross_attn_norm=True, qk_norm="rms_norm_across_heads",
        rope_max_seq_len=32, vace_layers=[0, 2], vace_in_channels=96,
    )
    return WanVACEPipeline(tokenizer=None, text_encoder=None, vae=vae,
                           scheduler=FlowMatchEulerDiscreteScheduler(shift=3.0), transformer=transformer)


def synthetic_inputs():
    """Deterministic reference (colour gradient) and control video (a square moving right)."""
    yy, xx = np.mgrid[0:H, 0:W]
    ref = np.stack([xx * 255 // W, yy * 255 // H, np.full_like(xx, 128)], -1).astype(np.uint8)
    control = []
    for t in range(FRAMES):
        f = np.zeros((H, W, 3), np.uint8)
        f[10:20, 4 + 3 * t:14 + 3 * t] = 255
        control.append(Image.fromarray(f))
    g = torch.Generator().manual_seed(SEED)
    pe = torch.randn(1, 8, 32, generator=g)
    ne = torch.zeros(1, 8, 32)
    return Image.fromarray(ref), control, pe, ne


def frames_digest(frames) -> dict:
    arr = np.asarray(frames, dtype=np.float32)
    return {"shape": list(arr.shape), "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
            "finite": bool(np.isfinite(arr).all()), "min": float(arr.min()), "max": float(arr.max()),
            "mean": float(arr.mean())}


def main() -> int:
    from common.env import detect_hardware
    from common.provenance import environment_fingerprint
    from inference.baseline_vace import BaselineSettings, generate

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "benchmark/baseline/tiny_vace_cpu.json"))
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    s = BaselineSettings(height=H, width=W, num_frames=FRAMES, num_inference_steps=STEPS, dtype="float32", seed=SEED)
    runs = []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        pipe = tiny_vace_pipe()
        t_build = time.perf_counter() - t0
        ref, control, pe, ne = synthetic_inputs()
        frames, stats = generate(pipe, s, ref, control, pe, ne)
        runs.append({"build_s": round(t_build, 4), "total_s": round(time.perf_counter() - t0, 4),
                     "generation_time_s": stats["generation_time_s"], "vram_peak_gb": stats.get("vram_peak_gb"),
                     "output": frames_digest(frames)})
    digests = {r["output"]["sha256"] for r in runs}
    result = {
        "kind": "tiny-vace-cpu-baseline",
        "code_path": "inference.baseline_vace.generate (pre-refactor)",
        "settings": {"height": H, "width": W, "num_frames": FRAMES, "steps": STEPS, "seed": SEED, "dtype": "float32"},
        "deterministic_across_repeats": len(digests) == 1,
        "runs": runs,
        "hardware": detect_hardware().to_dict(),
        "environment": environment_fingerprint(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("deterministic_across_repeats",)} | {"sha256": sorted(digests),
                     "total_s": [r["total_s"] for r in runs]}, indent=2))
    return 0 if len(digests) == 1 and all(r["output"]["finite"] for r in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
