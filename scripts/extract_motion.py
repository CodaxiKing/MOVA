"""Extract body / face / hand motion from a video (runs on CPU).

Usage:
  python scripts/extract_motion.py --video assets/motion/dance.mp4 --out outputs/motion/dance
  python scripts/extract_motion.py --video v.mp4 --out out/ --set max_frames=64 --set face=false
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_config, resolve_path  # noqa: E402
from common.experiment import ExperimentRun  # noqa: E402
from preprocessing.pipeline import ExtractionConfig, extract_motion  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/extraction.yaml")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. --set max_frames=32")
    args = ap.parse_args()

    raw = load_config(args.config, args.set)
    known = {f.name for f in fields(ExtractionConfig)}
    unknown = set(raw) - known
    if unknown:
        raise SystemExit(f"Unknown extraction config keys: {sorted(unknown)}")
    cfg = ExtractionConfig(**raw)

    video = resolve_path(args.video)
    if not video.exists():
        raise SystemExit(f"Video not found: {video}")
    run = ExperimentRun("extract")
    run.log(dataset=str(args.video), model="mediapipe", config=raw, output_dir=str(resolve_path(args.out)))
    try:
        summary = extract_motion(video, resolve_path(args.out), cfg)
    except Exception as e:
        run.finish("failed", error=repr(e))
        raise
    run.finish("success", frames=summary["num_frames"], resolution=f"{summary['width']}x{summary['height']}",
               summary=summary)
    print(f"Run record: {run.dir / 'run.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
