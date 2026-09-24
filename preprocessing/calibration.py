"""Sweep MediaPipe `min_confidence` on a video: detection rate vs jitter per stream (EXP-002 tool).

The threshold trades missed detections (too high) against unstable, hallucinated keypoints (too low). This only
measures; it recommends the highest-coverage threshold whose body jitter stays within `jitter_tolerance` of the
smoothest one (ties: the smoother one), and says so. Calibrate on real footage of the target domain
before changing the default (0.5).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def sweep_min_confidence(video: str | Path, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7), *, max_frames: int | None = None,
                         jitter_tolerance: float = 0.2) -> dict[str, Any]:
    import torch

    from preprocessing.features import temporal_jitter
    from preprocessing.pipeline import ExtractionConfig, extract_motion

    rows = []
    with tempfile.TemporaryDirectory(prefix="mova-calib-") as tmp:
        for thr in thresholds:
            out = Path(tmp) / f"t{thr}"
            summary = extract_motion(video, out, ExtractionConfig(min_confidence=float(thr), max_frames=max_frames,
                                                                  write_previews=False, write_openpose=False))
            hands = torch.load(out / "hand_motion.pt", weights_only=True)
            kp, pres = hands["kp2d"].numpy(), hands["present"].numpy()
            hand_jit = [temporal_jitter(kp[:, s, :, :2], pres[:, s]) if pres[:, s].sum() >= 3 else None for s in (0, 1)]
            rows.append({"min_confidence": float(thr), "body_detection": summary["body"]["detection_rate"],
                         "body_jitter": summary["body"]["jitter_xy"], "face_detection": summary["face"]["detection_rate"],
                         "hand_detection_left": summary["hands"]["detection_rate_left"],
                         "hand_detection_right": summary["hands"]["detection_rate_right"],
                         "hand_jitter_left": hand_jit[0], "hand_jitter_right": hand_jit[1],
                         "hands_postprocess": summary.get("hands_postprocess")})
    detected = [r for r in rows if r["body_detection"] > 0]
    rec = None
    if detected:
        best_jit = min(r["body_jitter"] for r in detected)
        ok = [r for r in detected if r["body_jitter"] <= best_jit * (1 + jitter_tolerance) + 1e-12]
        rec = max(ok, key=lambda r: (r["body_detection"] + r["face_detection"], -r["body_jitter"]))["min_confidence"]
    return {"video": str(video), "rows": rows, "recommended_min_confidence": rec,
            "rule": f"max coverage with body jitter <= best * (1 + {jitter_tolerance})",
            "note": "Measured on this video only; change configs/extraction.yaml only with real footage (EXP-002)."}
