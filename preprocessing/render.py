"""Rendering of keypoint tracks into preview frames and into OpenPose-style control videos."""

from __future__ import annotations

import colorsys

import cv2
import numpy as np

from .topology import MP_HAND_EDGES, MP_POSE_EDGES, OPENPOSE18_COLORS, OPENPOSE18_FROM_MP, OPENPOSE18_LIMBS


def _px(pt: np.ndarray, w: int, h: int) -> tuple[int, int]:
    return int(round(float(pt[0]) * w)), int(round(float(pt[1]) * h))


def blank(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def body_xyv(kp2d: np.ndarray) -> np.ndarray:
    """Stored body rows are [x, y, z, visibility]; drawing functions expect [x, y, visibility]."""
    if kp2d.shape[-1] != 4:
        raise ValueError(f"Expected (..., 4) body keypoints, got {kp2d.shape}")
    return kp2d[..., [0, 1, 3]]


def mp_body_to_openpose18(kp: np.ndarray) -> np.ndarray:
    """(33, 3[x, y, vis]) MediaPipe -> (18, 3) OpenPose ordering; neck = shoulder midpoint."""
    out = np.zeros((18, 3), dtype=np.float32)
    for i, src in enumerate(OPENPOSE18_FROM_MP):
        if src == "neck":
            out[i, :2] = (kp[11, :2] + kp[12, :2]) / 2.0
            out[i, 2] = min(kp[11, 2], kp[12, 2])
        else:
            out[i] = kp[int(src)]
    return out


def draw_openpose_body(img: np.ndarray, kp18: np.ndarray, vis_thr: float = 0.5, stick: int | None = None) -> np.ndarray:
    h, w = img.shape[:2]
    stick = stick or max(2, int(round(min(h, w) / 128)))
    for i, (a, b) in enumerate(OPENPOSE18_LIMBS):
        if kp18[a, 2] < vis_thr or kp18[b, 2] < vis_thr:
            continue
        color = tuple(int(c * 0.6) for c in OPENPOSE18_COLORS[i])
        cv2.line(img, _px(kp18[a], w, h), _px(kp18[b], w, h), color, stick * 2, cv2.LINE_AA)
    for i in range(18):
        if kp18[i, 2] >= vis_thr:
            cv2.circle(img, _px(kp18[i], w, h), stick + 1, OPENPOSE18_COLORS[i], -1, cv2.LINE_AA)
    return img


def draw_openpose_hand(img: np.ndarray, kp21: np.ndarray, thickness: int | None = None) -> np.ndarray:
    h, w = img.shape[:2]
    thickness = thickness or max(1, int(round(min(h, w) / 256)))
    for i, (a, b) in enumerate(MP_HAND_EDGES):
        r, g, bl = colorsys.hsv_to_rgb(i / len(MP_HAND_EDGES), 1.0, 1.0)
        cv2.line(img, _px(kp21[a], w, h), _px(kp21[b], w, h), (int(r * 255), int(g * 255), int(bl * 255)),
                 thickness * 2, cv2.LINE_AA)
    for p in kp21:
        cv2.circle(img, _px(p, w, h), thickness + 1, (0, 0, 255), -1, cv2.LINE_AA)
    return img


def draw_face_points(img: np.ndarray, kp: np.ndarray, color=(255, 255, 255), radius: int = 1, step: int = 1) -> np.ndarray:
    h, w = img.shape[:2]
    for p in kp[::step]:
        cv2.circle(img, _px(p, w, h), radius, color, -1)
    return img


def draw_mp_body(img: np.ndarray, kp: np.ndarray, vis_thr: float = 0.5) -> np.ndarray:
    """Debug preview in native MediaPipe topology (all 33 points)."""
    h, w = img.shape[:2]
    t = max(1, int(round(min(h, w) / 200)))
    for a, b in MP_POSE_EDGES:
        if kp[a, 2] >= vis_thr and kp[b, 2] >= vis_thr:
            cv2.line(img, _px(kp[a], w, h), _px(kp[b], w, h), (0, 255, 0), t * 2, cv2.LINE_AA)
    for p in kp:
        if p[2] >= vis_thr:
            cv2.circle(img, _px(p, w, h), t + 1, (255, 80, 80), -1, cv2.LINE_AA)
    return img


def overlay(frame: np.ndarray, layer: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    mask = layer.any(axis=-1, keepdims=True)
    blended = (frame * (1 - alpha) + layer * alpha).astype(np.uint8)
    return np.where(mask, blended, frame)


def put_label(img: np.ndarray, text: str) -> np.ndarray:
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img
