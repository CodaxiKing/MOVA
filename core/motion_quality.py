"""Cheap visibility diagnostics for a driving video's extracted body track."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def full_body_coverage(track: dict, *, threshold: float = 0.5) -> dict:
    """Count frames with both ankles detected inside the image.

    MediaPipe may extrapolate coordinates beyond the crop. Those points must not
    be treated as observed full-body motion even when the torso is detected.
    """
    kp = np.asarray(track["kp2d"])
    present = np.asarray(track["present"], dtype=bool)
    ankles = kp[:, [27, 28], :]
    visible = (ankles[:, :, 3] >= threshold) & (ankles[:, :, :2] >= 0).all(axis=2) & (ankles[:, :, :2] <= 1).all(axis=2)
    good = present & visible.all(axis=1)
    return {"frames": int(len(good)), "full_body_frames": int(good.sum()),
            "full_body_fraction": round(float(good.mean()), 3) if len(good) else 0.0}


def inspect_motion_track(extraction_dir: Path) -> dict | None:
    path = extraction_dir / "body_motion.pt"
    if not path.is_file():
        return None
    import torch

    return full_body_coverage(torch.load(path, map_location="cpu", weights_only=False))
