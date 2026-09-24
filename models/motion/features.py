"""Model inputs from the saved motion tracks (docs/pipeline.md, FORMAT_VERSION 1 and 2). NumPy only.

Per frame:
  body  (33, 6): hip-centred / torso-scaled 2D xy (2) + world xyz in metres (3) + visibility (1)
  face  (58,)  : 52 blendshapes + head rotation in 6D (Zhou et al.)
  hands (2, 21, 4): wrist-centred / palm-scaled world xyz (3) + presence (1), person's left then right
Each stream has a per-frame presence mask. Absent frames are zero-filled and masked, never interpolated here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from preprocessing.features import normalize_body, normalize_hand, rotation_matrix_to_6d

BODY_DIM, FACE_DIM, HAND_DIM = 6, 58, 4


def _np(x) -> np.ndarray:
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def body_inputs(body: dict[str, Any], visibility: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    kp = _np(body["kp2d"]).astype(np.float32)
    world = _np(body["kp3d_world"]).astype(np.float32)
    present = _np(body["present"]).astype(bool)
    anchors_ok = (kp[:, [11, 12, 23, 24], 3] >= visibility).all(-1)
    norm, _, _ = normalize_body(kp[..., :2])
    ok = present & anchors_ok & np.isfinite(norm).all((-1, -2)) & np.isfinite(world).all((-1, -2))
    x = np.concatenate([norm, world, kp[..., 3:4]], -1)
    x[~ok] = 0
    return x.astype(np.float32), ok


def face_inputs(face: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    bs = _np(face["blendshapes"]).astype(np.float32)
    rot = rotation_matrix_to_6d(_np(face["head_transform"])[:, :3, :3].astype(np.float32))
    present = _np(face["present"]).astype(bool)
    x = np.concatenate([bs, rot], -1)
    ok = present & np.isfinite(x).all(-1)
    x[~ok] = 0
    return x.astype(np.float32), ok


def hand_inputs(hands: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    world = _np(hands["kp3d_world"]).astype(np.float32)
    present = _np(hands["present"]).astype(bool)
    t = len(world)
    x = np.zeros((t, 2, 21, HAND_DIM), np.float32)
    for side in (0, 1):
        x[:, side, :, :3] = normalize_hand(world[:, side])
    ok = present & np.isfinite(x).all((-1, -2))
    x[..., 3] = ok[..., None]
    x[~ok] = 0
    return x, ok


def tracks_to_inputs(tracks: dict[str, dict[str, Any]], start: int = 0, num_frames: int | None = None) -> dict[str, np.ndarray]:
    """Slice [start, start + num_frames) of every stream. Keys: body, body_mask, face, face_mask, hands, hands_mask."""
    b, bm = body_inputs(tracks["body"])
    f, fm = face_inputs(tracks["face"])
    h, hm = hand_inputs(tracks["hands"])
    n = len(b)
    if not (len(f) == len(h) == n):
        raise ValueError("Tracks have different frame counts")
    stop = n if num_frames is None else start + num_frames
    if start < 0 or stop > n:
        raise ValueError(f"Window [{start}, {stop}) outside the {n} tracked frames")
    return {"body": b[start:stop], "body_mask": bm[start:stop], "face": f[start:stop], "face_mask": fm[start:stop],
            "hands": h[start:stop], "hands_mask": hm[start:stop]}
