"""Motion representations derived from raw keypoint tracks.

All functions operate on arrays shaped (T, K, D) with an optional presence mask (T,) or (T, K).
These are the candidate representations compared in EXP-003.
"""

from __future__ import annotations

import numpy as np

from .topology import LEFT_HIP, LEFT_SHOULDER, RIGHT_HIP, RIGHT_SHOULDER


def interpolate_missing(x: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Linearly fill frames where present==False; edges are held constant. x: (T, ...)."""
    x = np.array(x, dtype=np.float32, copy=True)
    present = np.asarray(present, dtype=bool)
    if present.all() or not present.any():
        return x
    t = np.arange(len(x))
    flat = x.reshape(len(x), -1)
    for j in range(flat.shape[1]):
        flat[~present, j] = np.interp(t[~present], t[present], flat[present, j])
    return flat.reshape(x.shape)


def ema_smooth(x: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Causal exponential moving average over time. alpha=1 -> no smoothing."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    out = np.array(x, dtype=np.float32, copy=True)
    for t in range(1, len(out)):
        out[t] = alpha * out[t] + (1.0 - alpha) * out[t - 1]
    return out


def velocity(x: np.ndarray, fps: float) -> np.ndarray:
    """First temporal difference in units/second; v[0] = 0."""
    v = np.zeros_like(x, dtype=np.float32)
    v[1:] = (x[1:] - x[:-1]) * fps
    return v


def acceleration(x: np.ndarray, fps: float) -> np.ndarray:
    return velocity(velocity(x, fps), fps)


def normalize_body(kp: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center on the hip midpoint and scale by torso length (shoulder-mid to hip-mid), per frame.

    Removes the driver's position and overall size so motion can be retargeted to a character with
    different framing. Returns (normalized, center (T, D), scale (T,)).
    """
    hip_mid = (kp[:, LEFT_HIP] + kp[:, RIGHT_HIP]) / 2.0
    sh_mid = (kp[:, LEFT_SHOULDER] + kp[:, RIGHT_SHOULDER]) / 2.0
    scale = np.linalg.norm(sh_mid - hip_mid, axis=-1)
    scale = np.maximum(scale, eps)
    norm = (kp - hip_mid[:, None, :]) / scale[:, None, None]
    return norm.astype(np.float32), hip_mid.astype(np.float32), scale.astype(np.float32)


def normalize_hand(kp: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Center on the wrist (landmark 0) and scale by wrist->middle-MCP (landmark 9) distance."""
    wrist = kp[:, 0:1]
    scale = np.linalg.norm(kp[:, 9] - kp[:, 0], axis=-1)
    return ((kp - wrist) / np.maximum(scale, eps)[:, None, None]).astype(np.float32)


def rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
    """(…, 3, 3) rotation -> (…, 3) [pitch, yaw, roll] in degrees (x-y-z convention)."""
    sy = np.sqrt(R[..., 0, 0] ** 2 + R[..., 1, 0] ** 2)
    singular = sy < 1e-6
    pitch = np.where(singular, np.arctan2(-R[..., 1, 2], R[..., 1, 1]), np.arctan2(R[..., 2, 1], R[..., 2, 2]))
    yaw = np.arctan2(-R[..., 2, 0], sy)
    roll = np.where(singular, 0.0, np.arctan2(R[..., 1, 0], R[..., 0, 0]))
    return np.degrees(np.stack([pitch, yaw, roll], axis=-1)).astype(np.float32)


def rotation_matrix_to_6d(R: np.ndarray) -> np.ndarray:
    """Continuous 6D rotation representation (Zhou et al. 2019): first two columns flattened."""
    return R[..., :, :2].reshape(*R.shape[:-2], 6).astype(np.float32)


def temporal_jitter(x: np.ndarray, present: np.ndarray | None = None) -> float:
    """Mean magnitude of second difference — a simple smoothness/jitter score (lower = smoother)."""
    if len(x) < 3:
        return 0.0
    acc = x[2:] - 2 * x[1:-1] + x[:-2]
    mag = np.linalg.norm(acc.reshape(len(acc), -1, acc.shape[-1]), axis=-1)
    if present is not None:
        m = np.asarray(present, bool)
        valid = m[2:] & m[1:-1] & m[:-2]
        if not valid.any():
            return 0.0
        mag = mag[valid]
    return float(mag.mean())
