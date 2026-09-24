"""Phase 1 baseline: reference image + motion video -> generated video with Wan2.1-VACE-1.3B.

Kept for compatibility (benchmark generate calls it). Equivalent to `mova infer`; the logic lives in
core/inference.py. Accepts every `mova infer` flag (--model, --runtime, --device, --precision, --offload, ...).

Safety gates:
  * Refuses to download model weights unless --allow-download is given (prints size + license first).
  * Refuses to run the 1.3B model on CPU unless --allow-cpu is given.

Usage:
  python scripts/check_model_size.py Wan-AI/Wan2.1-VACE-1.3B-diffusers
  python scripts/inference_baseline.py --config configs/baseline.yaml --allow-download
  python scripts/inference_baseline.py --set generation.num_frames=9 --set generation.height=256
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mova.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["infer", *sys.argv[1:]]))
