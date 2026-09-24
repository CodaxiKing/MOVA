"""Temporal consistency ("Dynamic Quality") diagnostics from optical flow. CPU, OpenCV only.

temporal-v1, per consecutive frame pair (t, t+1):
- warp_error: frame t is warped onto t+1 with backward Farneback flow; mean |RGB difference| (0..1) over pixels
  that pass a forward-backward consistency check (occlusions are excluded, reported as valid_fraction).
  Texture flicker, boiling and popping raise it; smooth motion does not.
- static_flicker: mean |I(t+1) - I(t)| over pixels whose flow is < 0.5 px (things that should not change).
- luma_flicker: mean |second difference| of the frame-mean luminance (global brightness pumping).
- mean_flow_px: motion magnitude. A frozen video has near-zero warp error AND near-zero flow: always read both.

Keypoint acceleration (evaluation/motion.py) cannot see any of this: the skeleton can be perfect while the
texture boils. When the driver video is given, the same numbers are computed on it (a real video of the same
motion) so the generated values have a reference point: `*_ratio` = generated / driver.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

TEMPORAL_VERSION = "temporal-v1"
STATIC_FLOW_PX = 0.5
_FB = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)


def _gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(frame, dtype=np.uint8), cv2.COLOR_RGB2GRAY)


def _flow(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Flow f such that a(x) ~ b(x + f(x))."""
    return cv2.calcOpticalFlowFarneback(a, b, None, **_FB)


def _warp(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def pair_metrics(prev: np.ndarray, nxt: np.ndarray) -> dict[str, float]:
    g0, g1 = _gray(prev), _gray(nxt)
    back = _flow(g1, g0)   # nxt(x) ~ prev(x + back(x))
    fwd = _flow(g0, g1)
    fwd_at = _warp(fwd, back)
    # Forward-backward consistency (Sundaram et al. 2010): occluded / unreliable pixels are excluded.
    mag = (back ** 2).sum(-1) + (fwd_at ** 2).sum(-1)
    valid = ((back + fwd_at) ** 2).sum(-1) < 0.01 * mag + 0.5
    warped = _warp(prev.astype(np.float32) / 255, back)
    diff = np.abs(nxt.astype(np.float32) / 255 - warped).mean(-1)
    flow_mag = np.linalg.norm(back, axis=-1)
    static = flow_mag < STATIC_FLOW_PX
    raw = np.abs(nxt.astype(np.float32) - prev.astype(np.float32)).mean(-1) / 255
    return {"warp_error": float(diff[valid].mean()) if valid.any() else float("nan"),
            "valid_fraction": float(valid.mean()),
            "static_flicker": float(raw[static].mean()) if static.any() else float("nan"),
            "static_fraction": float(static.mean()),
            "mean_flow_px": float(flow_mag.mean())}


def _nanmean(values) -> float | None:
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else None


def video_temporal_stats(frames: list[np.ndarray]) -> dict[str, Any]:
    if len(frames) < 2:
        return {"pairs": 0, "warp_error": None, "warp_error_max": None, "static_flicker": None, "luma_flicker": None,
                "mean_flow_px": None, "valid_fraction": None}
    pairs = [pair_metrics(frames[i], frames[i + 1]) for i in range(len(frames) - 1)]
    luma = np.array([_gray(f).mean() / 255 for f in frames])
    warp = [p["warp_error"] for p in pairs]
    finite = [w for w in warp if np.isfinite(w)]
    return {
        "pairs": len(pairs),
        "warp_error": _nanmean(warp),
        "warp_error_max": float(max(finite)) if finite else None,
        "static_flicker": _nanmean([p["static_flicker"] for p in pairs]),
        "luma_flicker": float(np.abs(np.diff(luma, n=2)).mean()) if len(luma) >= 3 else None,
        "mean_flow_px": _nanmean([p["mean_flow_px"] for p in pairs]),
        "valid_fraction": _nanmean([p["valid_fraction"] for p in pairs]),
        "per_pair_warp_error": [None if not np.isfinite(w) else round(float(w), 6) for w in warp],
    }


def _ratio(a, b):
    return None if a is None or b is None or b <= 1e-9 else float(a / b)


def temporal_metrics(generated: list[np.ndarray], driver: list[np.ndarray] | None = None) -> dict[str, Any]:
    gen = video_temporal_stats(generated)
    out: dict[str, Any] = {"metric_version": TEMPORAL_VERSION, "generated": gen}
    if driver:
        n = min(len(generated), len(driver))
        h, w = generated[0].shape[:2]
        drv = [cv2.resize(np.ascontiguousarray(f), (w, h), interpolation=cv2.INTER_AREA) for f in driver[:n]]
        d = video_temporal_stats(drv)
        out["driver"] = {k: v for k, v in d.items() if k != "per_pair_warp_error"}
        for key in ("warp_error", "static_flicker", "luma_flicker", "mean_flow_px"):
            out[f"{key}_ratio"] = _ratio(gen[key], d[key])
    return out
