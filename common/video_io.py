"""Video reading/writing. Frames are RGB uint8 numpy arrays of shape (H, W, 3)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np


@dataclass
class VideoMeta:
    fps: float
    width: int
    height: int
    num_frames: int


def probe_video(path: str | Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    meta = VideoMeta(
        fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        num_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return meta


def iter_frames(path: str | Path, target_fps: float | None = None, max_frames: int | None = None,
                ) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (index, timestamp_ms, rgb_frame), optionally resampled to target_fps."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = 1.0 if not target_fps or target_fps >= src_fps else src_fps / target_fps
    next_pick, src_idx, out_idx = 0.0, 0, 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if src_idx >= next_pick:
                yield out_idx, src_idx * 1000.0 / src_fps, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                out_idx += 1
                next_pick += step
                if max_frames is not None and out_idx >= max_frames:
                    break
            src_idx += 1
    finally:
        cap.release()


def read_video(path: str | Path, target_fps: float | None = None, max_frames: int | None = None) -> list[np.ndarray]:
    return [f for _, _, f in iter_frames(path, target_fps, max_frames)]


def write_video(path: str | Path, frames: Sequence[np.ndarray], fps: float) -> Path:
    """Write H.264 mp4 via imageio-ffmpeg (bundled ffmpeg binary)."""
    import imageio.v2 as imageio

    if len(frames) == 0:
        raise ValueError("No frames to write")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8, macro_block_size=1) as w:
        for f in frames:
            w.append_data(np.ascontiguousarray(f))
    return path
