"""Body motion: MediaPipe Pose Landmarker (33 landmarks, image-normalized 2D + metric world 3D)."""

from __future__ import annotations

from typing import Any

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from preprocessing.mp_models import ensure_model
from preprocessing.topology import MP_POSE_NAMES

NUM = len(MP_POSE_NAMES)


class BodyExtractor:
    def __init__(self, model: str = "pose_landmarker_full", min_confidence: float = 0.5) -> None:
        opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(model))),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_confidence,
            min_pose_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._lm = vision.PoseLandmarker.create_from_options(opts)
        self.model = model
        self.kp2d: list[np.ndarray] = []   # (33, 4): x, y (normalized [0,1]), z (relative depth), visibility
        self.kp3d: list[np.ndarray] = []   # (33, 3): world coords in meters, hip-centered
        self.present: list[bool] = []

    def process(self, rgb: np.ndarray, timestamp_ms: int) -> np.ndarray:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self._lm.detect_for_video(image, int(timestamp_ms))
        if res.pose_landmarks:
            lms = res.pose_landmarks[0]
            k2 = np.array([[l.x, l.y, l.z, l.visibility or 0.0] for l in lms], dtype=np.float32)
            wl = res.pose_world_landmarks[0]
            k3 = np.array([[l.x, l.y, l.z] for l in wl], dtype=np.float32)
            self.present.append(True)
        else:
            k2 = np.zeros((NUM, 4), np.float32)
            k3 = np.zeros((NUM, 3), np.float32)
            self.present.append(False)
        self.kp2d.append(k2)
        self.kp3d.append(k3)
        return k2

    def result(self) -> dict[str, Any]:
        return {
            "kp2d": np.stack(self.kp2d) if self.kp2d else np.zeros((0, NUM, 4), np.float32),
            "kp3d_world": np.stack(self.kp3d) if self.kp3d else np.zeros((0, NUM, 3), np.float32),
            "present": np.array(self.present, dtype=bool),
            "landmark_names": MP_POSE_NAMES,
            "extractor": f"mediapipe:{self.model}",
        }

    def close(self) -> None:
        self._lm.close()
