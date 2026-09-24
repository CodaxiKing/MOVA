"""Output integrity checks, independent of model weights and CUDA."""

from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np


def prepare_generated_frames(frames, *, count: int, width: int, height: int):
    """Reject invalid model output before conversion can hide NaN/Inf."""
    if len(frames) != count:
        raise ValueError(f"Expected {count} frames, got {len(frames)}")
    result = []
    for index, frame in enumerate(frames):
        frame = np.asarray(frame)
        if frame.shape != (height, width, 3):
            raise ValueError(f"Frame {index}: unexpected shape {frame.shape}")
        if not np.isfinite(frame).all():
            raise ValueError(f"Frame {index}: NaN or Inf in generated output")
        if frame.dtype == np.uint8:
            result.append(frame)
        elif np.issubdtype(frame.dtype, np.floating) and np.all((frame >= 0) & (frame <= 1)):
            result.append((frame * 255).astype(np.uint8))
        else:
            raise ValueError(f"Frame {index}: expected uint8 or floats in [0, 1]")
    return result


def validate_video(path: str | Path, *, count: int, width: int, height: int, fps: float) -> dict:
    """Fully decode a constant-FPS output and compare it with its contract.

    FFmpeg error detection complements OpenCV, which can silently stop decoding.
    This checks file integrity, not visual quality or motion fidelity.
    """
    if count <= 0 or width <= 0 or height <= 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Expected video dimensions, count and FPS must be positive")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot decode output video: {path}")
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(actual_fps) or not np.isclose(actual_fps, fps, rtol=0, atol=0.01):
            raise ValueError(f"FPS mismatch: expected {fps}, got {actual_fps}")
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = ''.join(chr((fourcc >> (8 * i)) & 255) for i in range(4)).strip('\x00')
        if not codec:
            raise ValueError("Output codec could not be identified")
        decoded = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape != (height, width, 3):
                raise ValueError(f"Decoded frame {decoded}: unexpected shape {frame.shape}")
            decoded += 1
        if decoded != count:
            raise ValueError(f"Decoded frame count mismatch: expected {count}, got {decoded}")
    finally:
        cap.release()
    check = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-xerror', '-err_detect', 'explode',
         '-i', str(Path(path).resolve()), '-map', '0:v:0', '-f', 'null', '-'],
        capture_output=True, text=True, timeout=120,
    )
    if check.returncode or check.stderr.strip():
        raise ValueError(f"FFmpeg output validation failed: {check.stderr[-2000:]}")
    return dict(status="PASS", num_frames=decoded, width=width, height=height,
                fps=actual_fps, codec=codec, duration_s=decoded / actual_fps)
