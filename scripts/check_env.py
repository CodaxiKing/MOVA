"""Report Python / PyTorch / CUDA / GPU / VRAM / RAM and the auto-selected runtime profile.

Usage: python scripts/check_env.py [--cuda-test] [--json outputs/env_report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from common.env import detect_hardware, select_profile, summarize  # noqa: E402


def cuda_smoke_test() -> str:
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    y = (x @ x).float().sum().item()
    torch.cuda.synchronize()
    return f"fp16 matmul OK (checksum {y:.3e}), peak {torch.cuda.max_memory_allocated() / 1024**2:.1f} MiB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda-test", action="store_true", help="run a small matmul on the GPU")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    hw = detect_hardware()
    profile = select_profile(hw)
    print(summarize(hw, profile))
    report = {"hardware": hw.to_dict(), "profile": profile.to_dict()}

    if args.cuda_test:
        if not hw.cuda_available:
            print("CUDA test    : SKIPPED (no CUDA device)")
            report["cuda_test"] = "skipped"
        else:
            msg = cuda_smoke_test()
            print(f"CUDA test    : {msg}")
            report["cuda_test"] = msg

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report saved : {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
