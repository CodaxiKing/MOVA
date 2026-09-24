"""Small MediaPipe Tasks model files (Apache 2.0, a few MB each), fetched from Google's official bucket."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from common.config import PROJECT_ROOT
from common.logging_utils import get_logger

log = get_logger("mova.mp_models")

MODEL_DIR = PROJECT_ROOT / "checkpoints" / "mediapipe"

MODELS: dict[str, str] = {
    "pose_landmarker_full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "pose_landmarker_heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "face_landmarker": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "hand_landmarker": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
}

MAX_BYTES = 50 * 1024**2  # refuse anything unexpectedly large


def ensure_model(name: str, model_dir: Path = MODEL_DIR) -> Path:
    if name not in MODELS:
        raise KeyError(f"Unknown MediaPipe model {name!r}; known: {sorted(MODELS)}")
    path = model_dir / f"{name}.task"
    if path.exists() and path.stat().st_size > 0:
        return path
    model_dir.mkdir(parents=True, exist_ok=True)
    url = MODELS[name]
    log.info("Downloading %s -> %s", url, path)
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=60) as resp:
        size = int(resp.headers.get("Content-Length") or 0)
        if size > MAX_BYTES:
            raise RuntimeError(f"{name} is {size / 1e6:.1f} MB, above the {MAX_BYTES / 1e6:.0f} MB safety limit")
        tmp.write_bytes(resp.read())
    tmp.replace(path)
    log.info("Saved %s (%.1f MB)", path.name, path.stat().st_size / 1e6)
    return path
