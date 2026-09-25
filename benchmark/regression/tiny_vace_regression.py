"""Regression: new architecture (model interface -> PyTorchRuntime) vs the Phase 0 baseline.

Feeds the exact synthetic inputs of benchmark/baseline/capture_tiny_vace.py through
models.registry -> WanVACETinyRandomModel -> runtime.get_runtime("pytorch") and requires a bit-identical
output (fp32, CPU). Also runs bf16/fp16 (finite output only) and records timings and memory.

Usage: python benchmark/regression/tiny_vace_regression.py [--repeat 3] [--out benchmark/regression/tiny_vace_cpu.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmark/baseline"))

from capture_tiny_vace import FRAMES, H, SEED, STEPS, W, frames_digest, synthetic_inputs  # noqa: E402
from reference import baseline_file  # noqa: E402

BASELINE = baseline_file()  # per CPU instruction set, see benchmark/baseline/reference.py


def run_once(precision: str) -> dict:
    from common.config import load_config
    from models.registry import create_model
    from runtime import get_runtime

    t0 = time.perf_counter()
    rt = get_runtime("pytorch")
    ctx = rt.context(device="cpu", precision=precision, offload="none")
    cfg = load_config("configs/smoke_tiny_vace.yaml")
    model = create_model("tiny", cfg)
    s = model.configure(ctx, (W, H))
    assert (s.height, s.width, s.num_frames, s.num_inference_steps, s.seed) == (H, W, FRAMES, STEPS, SEED)
    ref, control, _, _ = synthetic_inputs()  # prompt embeds: the model derives the same ones from the seed
    t_load = time.perf_counter()
    model.load(rt, ctx)
    load_s = time.perf_counter() - t_load
    frames, stats = model.generate(ref, control)
    model.unload()
    return {"precision": precision, "load_s": round(load_s, 4), "total_s": round(time.perf_counter() - t0, 4),
            "generation_time_s": stats["generation_time_s"], "memory": stats["memory"], "output": frames_digest(frames)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "benchmark/regression/tiny_vace_cpu.json"))
    args = ap.parse_args()
    from common.provenance import environment_fingerprint

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    expected = baseline["runs"][0]["output"]["sha256"]
    fp32 = [run_once("fp32") for _ in range(args.repeat)]
    others = [run_once(p) for p in ("bf16", "fp16")]
    bit_exact = all(r["output"]["sha256"] == expected for r in fp32)
    finite = all(r["output"]["finite"] for r in fp32 + others)
    base_t = [r["total_s"] for r in baseline["runs"][1:]] or [baseline["runs"][0]["total_s"]]
    new_t = [r["total_s"] for r in fp32[1:]] or [fp32[0]["total_s"]]
    result = {
        "kind": "tiny-vace-cpu-regression",
        "code_path": "models.registry -> WanVACETinyRandomModel -> PyTorchRuntime",
        "baseline_sha256": expected, "fp32_bit_exact": bit_exact, "all_finite": finite,
        "warm_total_s": {"baseline_median": sorted(base_t)[len(base_t) // 2], "new_median": sorted(new_t)[len(new_t) // 2]},
        "fp32_runs": fp32, "other_precisions": others, "environment": environment_fingerprint(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("fp32_bit_exact", "all_finite", "warm_total_s")}
                     | {"bf16_fp16_sha256": [r["output"]["sha256"][:12] for r in others]}, indent=2))
    return 0 if bit_exact and finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
