"""Temporal filters for keypoint tracks.

One-Euro filter (Casiez, Roussel & Vogel, CHI 2012): a low-pass filter whose cutoff rises with speed, so slow
motion is strongly de-jittered while fast motion keeps little lag. Per sample:

    dx      = (x_t - x̂_{t-1}) * fps ; dx̂ = lowpass(dx, d_cutoff)
    cutoff  = min_cutoff + beta * |dx̂|
    x̂_t     = lowpass(x_t, cutoff),  alpha(c) = 1 / (1 + fps / (2π c))

Offline over (T, ...) arrays with a presence mask: the state resets after a missing frame, so a detection
after a gap is never pulled towards stale data, and missing frames are never filled here.
"""

from __future__ import annotations

import numpy as np


def _alpha(cutoff: np.ndarray | float, fps: float) -> np.ndarray:
    tau = 1.0 / (2 * np.pi * np.asarray(cutoff, dtype=np.float64))
    return 1.0 / (1.0 + tau * fps)


def one_euro(x: np.ndarray, fps: float, *, min_cutoff: float = 1.0, beta: float = 0.5, d_cutoff: float = 1.0,
             present: np.ndarray | None = None) -> np.ndarray:
    """Filter x (T, ...) along time. present: (T,) or broadcastable to x[..., 0] (e.g. (T, K)); False = missing."""
    if fps <= 0 or min_cutoff <= 0 or d_cutoff <= 0 or beta < 0:
        raise ValueError("fps, min_cutoff, d_cutoff must be > 0 and beta >= 0")
    x = np.asarray(x, dtype=np.float64)
    t = len(x)
    if present is None:
        pres = np.ones(x.shape[:-1], bool)
    else:
        pres = np.broadcast_to(np.asarray(present, bool).reshape(np.asarray(present).shape + (1,) * (x.ndim - 1 - np.asarray(present).ndim)), x.shape[:-1])
    out = x.copy()
    prev = np.zeros(x.shape[1:])
    dprev = np.zeros(x.shape[1:])
    has = np.zeros(x.shape[1:-1], bool)          # per point: filter state is valid
    a_d = _alpha(d_cutoff, fps)
    for i in range(t):
        p = pres[i]
        fresh = p & ~has
        cont = p & has
        # new segment: start from the observation, zero derivative
        prev[fresh] = x[i][fresh]
        dprev[fresh] = 0.0
        if cont.any():
            dx = (x[i][cont] - prev[cont]) * fps
            dhat = a_d * dx + (1 - a_d) * dprev[cont]
            cutoff = min_cutoff + beta * np.linalg.norm(dhat, axis=-1, keepdims=True)
            a = _alpha(cutoff, fps)
            prev[cont] = a * x[i][cont] + (1 - a) * prev[cont]
            dprev[cont] = dhat
        out[i][p] = prev[p]
        has = p.copy()                            # a missing frame resets the state
    return out.astype(np.float32)


def jitter(x: np.ndarray, present: np.ndarray | None = None) -> float:
    """Mean |second difference| over frames where three consecutive samples exist."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 3:
        return 0.0
    acc = np.linalg.norm(x[2:] - 2 * x[1:-1] + x[:-2], axis=-1)
    if present is not None:
        m = np.asarray(present, bool)
        m = m[2:] & m[1:-1] & m[:-2]
        m = np.broadcast_to(m.reshape(m.shape + (1,) * (acc.ndim - m.ndim)), acc.shape)
        return float(acc[m].mean()) if m.any() else 0.0
    return float(acc.mean())
