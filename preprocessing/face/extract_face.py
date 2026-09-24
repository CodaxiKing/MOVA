"""Face motion: MediaPipe Face Landmarker (478 landmarks, 52 blendshapes, 4x4 head transform)."""

from __future__ import annotations

from typing import Any

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from preprocessing.features import rotation_matrix_to_euler
from preprocessing.mp_models import ensure_model
from preprocessing.topology import FACE_NUM_BLENDSHAPES, FACE_NUM_LANDMARKS


class FaceExtractor:
    def __init__(self, min_confidence: float = 0.5) -> None:
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model("face_landmarker"))),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._lm = vision.FaceLandmarker.create_from_options(opts)
        self.landmarks: list[np.ndarray] = []
        self.blendshapes: list[np.ndarray] = []
        self.transforms: list[np.ndarray] = []
        self.present: list[bool] = []
        self.blendshape_names: list[str] | None = None

    def process(self, rgb: np.ndarray, timestamp_ms: int) -> np.ndarray | None:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self._lm.detect_for_video(image, int(timestamp_ms))
        if res.face_landmarks:
            lm = np.array([[l.x, l.y, l.z] for l in res.face_landmarks[0]], dtype=np.float32)
            cats = res.face_blendshapes[0] if res.face_blendshapes else []
            bs = np.array([c.score for c in cats], dtype=np.float32)
            if self.blendshape_names is None and cats:
                self.blendshape_names = [c.category_name for c in cats]
            tf = (np.asarray(res.facial_transformation_matrixes[0], dtype=np.float32)
                  if res.facial_transformation_matrixes else np.eye(4, dtype=np.float32))
            if bs.shape[0] != FACE_NUM_BLENDSHAPES:
                bs = np.resize(bs, FACE_NUM_BLENDSHAPES) if bs.size else np.zeros(FACE_NUM_BLENDSHAPES, np.float32)
            self.present.append(True)
        else:
            lm = np.zeros((FACE_NUM_LANDMARKS, 3), np.float32)
            bs = np.zeros(FACE_NUM_BLENDSHAPES, np.float32)
            tf = np.eye(4, dtype=np.float32)
            self.present.append(False)
        self.landmarks.append(lm)
        self.blendshapes.append(bs)
        self.transforms.append(tf)
        return lm if self.present[-1] else None

    def result(self) -> dict[str, Any]:
        tfs = np.stack(self.transforms) if self.transforms else np.zeros((0, 4, 4), np.float32)
        return {
            "landmarks": np.stack(self.landmarks) if self.landmarks else np.zeros((0, FACE_NUM_LANDMARKS, 3), np.float32),
            "blendshapes": np.stack(self.blendshapes) if self.blendshapes else np.zeros((0, FACE_NUM_BLENDSHAPES), np.float32),
            "blendshape_names": self.blendshape_names or [],
            "head_transform": tfs,
            "head_euler_deg": rotation_matrix_to_euler(tfs[:, :3, :3]) if len(tfs) else np.zeros((0, 3), np.float32),
            "head_translation": tfs[:, :3, 3] if len(tfs) else np.zeros((0, 3), np.float32),
            "present": np.array(self.present, dtype=bool),
            "extractor": "mediapipe:face_landmarker",
        }

    def close(self) -> None:
        self._lm.close()
