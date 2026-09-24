"""Hand track post-processing. Raw detections are kept (kp2d_raw / present_raw); every change is counted.

1. reject_far_hands: a detected hand whose wrist lies farther than `max_forearms` forearm lengths from the body
   wrist of the same side is a false positive (background hand, the other person...) -> present = False.
2. fix_side_swaps: when the body wrists cannot decide (occluded or low visibility), per-frame handedness labels
   flip. Assign each detection to the side whose last known wrist position is closest (temporal continuity);
   frames with confident body wrists are trusted as they are.
3. fill_short_gaps: interior gaps of <= max_gap frames are linearly interpolated into a SEPARATE array with its own
   mask (`filled`). Only the control video uses it (no blinking hands); `present` still means "detected", so
   evaluation never counts invented frames as evidence.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from preprocessing.topology import LEFT_WRIST, RIGHT_WRIST

BODY_WRIST = (LEFT_WRIST, RIGHT_WRIST)
BODY_ELBOW = (13, 14)


def reject_far_hands(kp2d: np.ndarray, present: np.ndarray, body_kp: np.ndarray | None, width: int, height: int,
                     *, max_forearms: float = 1.0, visibility: float = 0.5) -> tuple[np.ndarray, int]:
    present = present.copy()
    if body_kp is None:
        return present, 0
    size = np.array([width, height])
    rejected = 0
    for side in (0, 1):
        wr, el = BODY_WRIST[side], BODY_ELBOW[side]
        ok = (body_kp[:, wr, 3] >= visibility) & (body_kp[:, el, 3] >= visibility) & present[:, side]
        forearm = np.linalg.norm((body_kp[:, wr, :2] - body_kp[:, el, :2]) * size, axis=-1)
        dist = np.linalg.norm((kp2d[:, side, 0, :2] - body_kp[:, wr, :2]) * size, axis=-1)
        bad = ok & (forearm > 1e-6) & (dist > max_forearms * forearm)
        present[bad, side] = False
        rejected += int(bad.sum())
    return present, rejected


def fix_side_swaps(kp2d: np.ndarray, present: np.ndarray, body_kp: np.ndarray | None, *, visibility: float = 0.5,
                   max_age: int = 8) -> tuple[np.ndarray, int]:
    """Return (source (T, 2) int: raw side index feeding each output side, -1 = none; number of swaps)."""
    t = len(kp2d)
    source = np.where(present, np.arange(2)[None], -1)
    trusted = np.zeros(t, bool)
    if body_kp is not None:
        trusted = (body_kp[:, BODY_WRIST[0], 3] >= visibility) & (body_kp[:, BODY_WRIST[1], 3] >= visibility)
    last: list[np.ndarray | None] = [None, None]
    age = [10**9, 10**9]
    swaps = 0
    for i in range(t):
        if not trusted[i] and present[i].any() and all(last[s] is not None and age[s] <= max_age for s in (0, 1)):
            def d(raw, side):
                return np.linalg.norm(kp2d[i, raw, 0, :2] - last[side])

            if present[i].all():
                if d(0, 1) + d(1, 0) < d(0, 0) + d(1, 1):
                    source[i] = [1, 0]
                    swaps += 1
            else:
                raw = int(np.argmax(present[i]))
                if d(raw, 1 - raw) < 0.5 * d(raw, raw):   # clearly the other hand: move it
                    source[i, 1 - raw], source[i, raw] = raw, -1
                    swaps += 1
        for side in (0, 1):
            if source[i, side] >= 0:
                last[side], age[side] = kp2d[i, source[i, side], 0, :2].copy(), 0
            else:
                age[side] += 1
    return source, swaps


def _gather(x: np.ndarray, source: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    for side in (0, 1):
        ok = source[:, side] >= 0
        out[ok, side] = x[np.flatnonzero(ok), source[ok, side]]
    return out


def fill_short_gaps(x: np.ndarray, present: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation of interior gaps <= max_gap along axis 0. Returns (filled array, filled-frame mask)."""
    x = np.array(x, dtype=np.float32, copy=True)
    present = np.asarray(present, bool)
    filled = np.zeros(len(x), bool)
    idx = np.flatnonzero(present)
    for a, b in zip(idx[:-1], idx[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            for k in range(1, gap + 1):
                w = k / (gap + 1)
                x[a + k] = (1 - w) * x[a] + w * x[b]
                filled[a + k] = True
    return x, filled


def postprocess_hands(hands: dict[str, Any], body_kp: np.ndarray | None, width: int, height: int, *,
                      max_gap: int = 3, max_forearms: float = 1.0) -> tuple[dict[str, Any], dict[str, int]]:
    """Return a new hands track (raw arrays kept under *_raw) and a report of what changed."""
    kp = np.asarray(hands["kp2d"], np.float32)
    k3 = np.asarray(hands["kp3d_world"], np.float32)
    pres = np.asarray(hands["present"], bool)
    pres_rej, rejected = reject_far_hands(kp, pres, body_kp, width, height, max_forearms=max_forearms)
    source, swaps = fix_side_swaps(kp, pres_rej, body_kp)
    kp_fix, k3_fix = _gather(kp, source), _gather(k3, source)
    pres_fix = source >= 0
    score = np.asarray(hands.get("handedness_score", np.zeros(pres.shape, np.float32)))
    filled_kp = kp_fix.copy()
    filled_mask = np.zeros_like(pres_fix)
    for s in (0, 1):
        filled_kp[:, s], filled_mask[:, s] = fill_short_gaps(kp_fix[:, s], pres_fix[:, s], max_gap)
    out = dict(hands)
    out.update(kp2d=kp_fix, kp3d_world=k3_fix, present=pres_fix, handedness_score=_gather(score, source),
               kp2d_raw=kp, present_raw=pres,
               kp2d_filled=filled_kp, filled=filled_mask)
    report = {"rejected_far_from_wrist": rejected, "side_swaps_fixed": swaps,
              "frames_filled_for_control": int(filled_mask.sum()), "max_gap": max_gap}
    return out, report
