"""Render the OpenPose-style control video from saved tracks (instead of live, during extraction).

Rendering from tracks is what lets the control be post-processed first: smoothed (preprocessing/filters.py),
cleaned (preprocessing/hands_post.py) or retargeted to another body (preprocessing/retarget.py). Drawing is the
same as the live path in preprocessing/pipeline.py (same functions, same inputs).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from preprocessing import render as R


def _np(x) -> np.ndarray:
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def render_control_frames(body: dict[str, Any] | None, hands: dict[str, Any] | None, face: dict[str, Any] | None,
                          width: int, height: int, *, face_step: int = 4) -> list[np.ndarray]:
    n = max(len(_np(t["present"])) for t in (body, hands, face) if t is not None)
    bk, bp = (_np(body["kp2d"]), _np(body["present"]).astype(bool)) if body is not None else (None, None)
    hk, hp = (_np(hands["kp2d"]), _np(hands["present"]).astype(bool)) if hands is not None else (None, None)
    fk, fp = (_np(face["landmarks"]), _np(face["present"]).astype(bool)) if face is not None else (None, None)
    frames = []
    for t in range(n):
        ctrl = R.blank(height, width)
        if bk is not None and bp[t]:
            R.draw_openpose_body(ctrl, R.mp_body_to_openpose18(R.body_xyv(bk[t])))
        if hk is not None:
            for s in range(2):
                if hp[t, s]:
                    R.draw_openpose_hand(ctrl, hk[t, s])
        if fk is not None and fp[t]:
            R.draw_face_points(ctrl, fk[t], step=face_step)
        frames.append(ctrl)
    return frames
