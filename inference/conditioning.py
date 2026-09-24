"""Preparation of reference image and control video for Wan-family pipelines (no model dependencies)."""

from __future__ import annotations

import numpy as np
from PIL import Image

VAE_TEMPORAL = 4
SPATIAL_MULTIPLE = 16  # VAE 8x * patch 2x


def valid_num_frames(n: int) -> int:
    """Largest 4k+1 <= n (Wan VAE temporal constraint); minimum 5."""
    if n < 5:
        raise ValueError(f"Need at least 5 frames, got {n}")
    return ((n - 1) // VAE_TEMPORAL) * VAE_TEMPORAL + 1


def round_to_multiple(x: int, m: int = SPATIAL_MULTIPLE) -> int:
    return max(m, int(round(x / m)) * m)


def fit_resolution(src_w: int, src_h: int, max_area: int, multiple: int = SPATIAL_MULTIPLE) -> tuple[int, int]:
    """Aspect-preserving (width, height) with area <= max_area, both multiples of `multiple`."""
    aspect = src_h / src_w
    h = int(np.sqrt(max_area * aspect)) // multiple * multiple
    w = int(np.sqrt(max_area / aspect)) // multiple * multiple
    return max(multiple, w), max(multiple, h)


def letterbox(img: Image.Image, width: int, height: int, fill=(255, 255, 255)) -> Image.Image:
    """Resize to fit inside (width, height) keeping aspect, pad the rest with `fill`."""
    img = img.convert("RGB")
    scale = min(width / img.width, height / img.height)
    nw, nh = max(1, round(img.width * scale)), max(1, round(img.height * scale))
    canvas = Image.new("RGB", (width, height), fill)
    canvas.paste(img.resize((nw, nh), Image.LANCZOS), ((width - nw) // 2, (height - nh) // 2))
    return canvas


def sample_indices(n_src: int, n_out: int, stride: int = 1, start: int = 0) -> list[int]:
    """Pick n_out frame indices with a fixed stride, clamped to the source length."""
    idx = [min(start + i * stride, n_src - 1) for i in range(n_out)]
    return idx


def prepare_control_frames(frames: list[np.ndarray], width: int, height: int, num_frames: int,
                           stride: int = 1) -> list[Image.Image]:
    if not frames:
        raise ValueError("Empty control video")
    idx = sample_indices(len(frames), num_frames, stride)
    return [letterbox(Image.fromarray(frames[i]), width, height, fill=(0, 0, 0)) for i in idx]
