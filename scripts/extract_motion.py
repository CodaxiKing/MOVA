"""Extract body / face / hand motion from a video (runs on CPU). Equivalent to `mova preprocess`.

Usage:
  python scripts/extract_motion.py --video assets/motion/dance.mp4 --out outputs/motion/dance
  python scripts/extract_motion.py --video v.mp4 --out out/ --set max_frames=64 --set face=false
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mova.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["preprocess", *sys.argv[1:]]))
