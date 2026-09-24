"""EXP-003 (synthetic part): compare body-motion representations. See evaluation/representations.py.

Usage: python scripts/exp003_representations.py [--seeds 0 1 2] [--out docs/experiments/EXP-003.results.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.representations import run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    runs = [run(seed=s) for s in args.seeds]
    names = list(runs[0]["results"])
    keys = ("nuisance_camera", "nuisance_body_size", "noise_jitter_ratio", "pca_components_95", "prediction_error")
    summary = {n: {k: sum(r["results"][n][k] for r in runs) / len(runs) for k in keys} | {"dims": runs[0]["results"][n]["dims"]}
               for n in names}
    report = {"experiment": "EXP-003-synthetic", "seeds": args.seeds, "config": {k: v for k, v in runs[0].items() if k != "results"},
              "mean_over_seeds": summary}
    print(f"{'representation':28s} {'camera':>7s} {'body':>7s} {'noise':>7s} {'pca95':>6s} {'dims':>5s} {'pred':>6s}")
    for n, v in summary.items():
        print(f"{n:28s} {v['nuisance_camera']:7.3f} {v['nuisance_body_size']:7.3f} {v['noise_jitter_ratio']:7.3f} "
              f"{v['pca_components_95']:6.1f} {v['dims']:5d} {v['prediction_error']:6.3f}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
