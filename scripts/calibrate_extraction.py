"""Sweep MediaPipe min_confidence on a video and report detection rate vs jitter (EXP-002 tool).

Usage: python scripts/calibrate_extraction.py --video assets/motion/dance.mp4 [--thresholds 0.3 0.5 0.7] [--out report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import resolve_path  # noqa: E402
from preprocessing.calibration import sweep_min_confidence  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    report = sweep_min_confidence(resolve_path(args.video), args.thresholds, max_frames=args.max_frames)
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
