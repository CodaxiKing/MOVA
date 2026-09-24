"""Hand motion: MediaPipe Hand Landmarker (21 landmarks per hand, 2D + world 3D).

Left/right assignment: when body wrists are available, each detected hand is matched to the nearest body
wrist (robust to MediaPipe's mirrored-handedness convention). Otherwise the handedness label is used,
swapped for non-mirrored footage (MediaPipe assumes selfie/mirrored input).
"""

from __future__ import annotations

from typing import Any

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from preprocessing.mp_models import ensure_model
from preprocessing.topology import LEFT_WRIST, RIGHT_WRIST

SIDES = ("left", "right")  # the person's own left/right


class HandExtractor:
    def __init__(self, min_confidence: float = 0.5, mirrored_input: bool = False, wrist_vis_thr: float = 0.3) -> None:
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model("hand_landmarker"))),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._lm = vision.HandLandmarker.create_from_options(opts)
        self.mirrored_input = mirrored_input
        self.wrist_vis_thr = wrist_vis_thr
        self.kp2d: list[np.ndarray] = []    # (2, 21, 3) x, y normalized, z relative
        self.kp3d: list[np.ndarray] = []    # (2, 21, 3) world meters
        self.present: list[np.ndarray] = []  # (2,)
        self.score: list[np.ndarray] = []    # (2,) handedness confidence

    def _side_from_label(self, label: str) -> int:
        is_left_label = label.lower() == "left"
        person_left = is_left_label if self.mirrored_input else not is_left_label
        return 0 if person_left else 1

    def process(self, rgb: np.ndarray, timestamp_ms: int, body_kp2d: np.ndarray | None = None) -> np.ndarray:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self._lm.detect_for_video(image, int(timestamp_ms))
        k2 = np.zeros((2, 21, 3), np.float32)
        k3 = np.zeros((2, 21, 3), np.float32)
        pres = np.zeros(2, bool)
        sc = np.zeros(2, np.float32)

        hands = []
        for i, lms in enumerate(res.hand_landmarks or []):
            a2 = np.array([[l.x, l.y, l.z] for l in lms], np.float32)
            a3 = np.array([[l.x, l.y, l.z] for l in res.hand_world_landmarks[i]], np.float32)
            cat = res.handedness[i][0] if res.handedness and res.handedness[i] else None
            hands.append((a2, a3, cat.category_name if cat else "", float(cat.score) if cat else 0.0))

        sides = self._assign_sides(hands, body_kp2d)
        for (a2, a3, _, s), side in zip(hands, sides):
            if side is None or pres[side]:
                continue
            k2[side], k3[side], pres[side], sc[side] = a2, a3, True, s

        self.kp2d.append(k2)
        self.kp3d.append(k3)
        self.present.append(pres)
        self.score.append(sc)
        return k2

    def _assign_sides(self, hands: list, body_kp2d: np.ndarray | None) -> list[int | None]:
        wrists_ok = (body_kp2d is not None and body_kp2d[LEFT_WRIST, 3] >= self.wrist_vis_thr
                     and body_kp2d[RIGHT_WRIST, 3] >= self.wrist_vis_thr)
        if not wrists_ok:
            return [self._side_from_label(h[2]) for h in hands]
        wrists = np.stack([body_kp2d[LEFT_WRIST, :2], body_kp2d[RIGHT_WRIST, :2]])
        dists = np.array([[np.linalg.norm(h[0][0, :2] - w) for w in wrists] for h in hands])
        if len(hands) == 2:
            direct = dists[0, 0] + dists[1, 1]
            swapped = dists[0, 1] + dists[1, 0]
            return [0, 1] if direct <= swapped else [1, 0]
        return [int(np.argmin(d)) for d in dists]

    def result(self) -> dict[str, Any]:
        n = len(self.kp2d)
        return {
            "kp2d": np.stack(self.kp2d) if n else np.zeros((0, 2, 21, 3), np.float32),
            "kp3d_world": np.stack(self.kp3d) if n else np.zeros((0, 2, 21, 3), np.float32),
            "present": np.stack(self.present) if n else np.zeros((0, 2), bool),
            "handedness_score": np.stack(self.score) if n else np.zeros((0, 2), np.float32),
            "sides": list(SIDES),
            "extractor": "mediapipe:hand_landmarker",
        }

    def close(self) -> None:
        self._lm.close()
