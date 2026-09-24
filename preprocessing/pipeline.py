"""Run body / face / hand extraction over a video in a single decode pass.

Outputs (in out_dir):
    body_motion.pt, face_motion.pt, hand_motion.pt      raw tracks + derived features (torch.save dicts)
    body_preview.mp4, face_preview.mp4, hands_preview.mp4  overlays on the source video
    pose_openpose.mp4                                     OpenPose-style control video (black background)
    summary.json                                          detection rates, jitter, timings
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common.logging_utils import get_logger
from common.video_io import iter_frames, probe_video, write_video
from preprocessing import features as F
from preprocessing import render as R
from preprocessing.hands_post import postprocess_hands

log = get_logger("mova.extract")

# 2: hand kp2d/present are post-processed by default (raw kept as kp2d_raw/present_raw), kp2d_filled/filled
#    added; optional features *_smoothed. See docs/pipeline.md.
FORMAT_VERSION = 2


@dataclass
class ExtractionConfig:
    body: bool = True
    face: bool = True
    hands: bool = True
    target_fps: float | None = None
    max_frames: int | None = None
    min_confidence: float = 0.5
    pose_model: str = "pose_landmarker_full"
    mirrored_input: bool = False
    write_previews: bool = True
    write_openpose: bool = True
    smooth_alpha: float = 1.0  # 1.0 = raw tracks; smoothing is stored separately from raw data
    # Control-video signal cleaning (raw tracks are always saved unchanged; see docs/pipeline.md).
    smoothing: str = "none"          # none | one_euro  (one_euro: preprocessing/filters.py)
    one_euro_min_cutoff: float = 2.0  # Hz; preset from docs/experiments/EXP-006.md (synthetic calibration)
    one_euro_beta: float = 100.0      # speed coefficient for normalised image coordinates
    hand_postprocess: bool = True     # reject far hands, fix L/R swaps, fill short gaps for the control only
    hand_max_gap: int = 3             # frames


def _to_torch(d: dict[str, Any]) -> dict[str, Any]:
    return {k: torch.from_numpy(np.ascontiguousarray(v)) if isinstance(v, np.ndarray) else v for k, v in d.items()}


def _body_features(body: dict[str, Any], fps: float, alpha: float) -> dict[str, Any]:
    kp = body["kp2d"]
    if len(kp) == 0:
        return {}
    present = body["present"]
    xy = F.interpolate_missing(kp[..., :2], present)
    w3 = F.interpolate_missing(body["kp3d_world"], present)
    if alpha < 1.0:
        xy, w3 = F.ema_smooth(xy, alpha), F.ema_smooth(w3, alpha)
    norm, center, scale = F.normalize_body(xy)
    return {
        "xy_filled": xy, "world_filled": w3, "xy_normalized": norm, "center": center, "scale": scale,
        "velocity_normalized": F.velocity(norm, fps), "acceleration_normalized": F.acceleration(norm, fps),
        "velocity_world": F.velocity(w3, fps),
    }


def _face_features(face: dict[str, Any], fps: float) -> dict[str, Any]:
    if len(face["landmarks"]) == 0:
        return {}
    present = face["present"]
    euler = F.interpolate_missing(face["head_euler_deg"], present)
    bs = F.interpolate_missing(face["blendshapes"], present)
    rot6d = F.rotation_matrix_to_6d(face["head_transform"][:, :3, :3])
    return {"head_euler_filled": euler, "head_rot6d": rot6d, "blendshapes_filled": bs,
            "blendshapes_velocity": F.velocity(bs, fps)}


def _hand_features(hands: dict[str, Any], fps: float) -> dict[str, Any]:
    if len(hands["kp2d"]) == 0:
        return {}
    out: dict[str, Any] = {}
    for i, side in enumerate(hands["sides"]):
        pres = hands["present"][:, i]
        xy = F.interpolate_missing(hands["kp2d"][:, i, :, :2], pres)
        w3 = F.interpolate_missing(hands["kp3d_world"][:, i], pres)
        out[f"{side}_xy_filled"] = xy
        out[f"{side}_world_normalized"] = F.normalize_hand(w3)
        out[f"{side}_velocity_world"] = F.velocity(w3, fps)
    return out


def _smooth(x: np.ndarray, present: np.ndarray, fps: float, cfg: ExtractionConfig) -> np.ndarray:
    from preprocessing.filters import one_euro

    return one_euro(x, fps, min_cutoff=cfg.one_euro_min_cutoff, beta=cfg.one_euro_beta, present=present)


def _control_frames(b, hd, f, width, height, fps, cfg: ExtractionConfig) -> list[np.ndarray]:
    """OpenPose-style control from the tracks: filled hand gaps and (optionally) One-Euro smoothed positions.
    With smoothing 'none' and no hand post-processing this draws exactly what the live renderer used to draw."""
    from preprocessing.control import render_control_frames

    body = hands = face = None
    if b is not None and len(b["kp2d"]):
        kp = b["kp2d"].copy()
        if cfg.smoothing == "one_euro":
            kp[..., :2] = _smooth(kp[..., :2], b["present"], fps, cfg)
        body = {"kp2d": kp, "present": b["present"]}
    if hd is not None and len(hd["kp2d"]):
        kp = hd.get("kp2d_filled", hd["kp2d"]).copy()
        pres = hd["present"] | hd.get("filled", np.zeros_like(hd["present"]))
        if cfg.smoothing == "one_euro":
            kp[..., :2] = _smooth(kp[..., :2], pres, fps, cfg)
        hands = {"kp2d": kp, "present": pres}
    if f is not None and len(f["landmarks"]):
        lm = f["landmarks"].copy()
        if cfg.smoothing == "one_euro":
            lm[..., :2] = _smooth(lm[..., :2], f["present"], fps, cfg)
        face = {"landmarks": lm, "present": f["present"]}
    if body is None and hands is None and face is None:
        return []
    return render_control_frames(body, hands, face, width, height)


def extract_motion(video: str | Path, out_dir: str | Path, cfg: ExtractionConfig | None = None) -> dict[str, Any]:
    from preprocessing.face.extract_face import FaceExtractor
    from preprocessing.hands.extract_hands import HandExtractor
    from preprocessing.pose.extract_body import BodyExtractor

    cfg = cfg or ExtractionConfig()
    video, out_dir = Path(video), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = probe_video(video)
    fps = cfg.target_fps if cfg.target_fps and cfg.target_fps < meta.fps else meta.fps
    log.info("Video %s: %dx%d @ %.2f fps, %d frames -> extracting at %.2f fps",
             video.name, meta.width, meta.height, meta.fps, meta.num_frames, fps)

    body = BodyExtractor(cfg.pose_model, cfg.min_confidence) if cfg.body or cfg.hands else None
    face = FaceExtractor(cfg.min_confidence) if cfg.face else None
    hands = HandExtractor(cfg.min_confidence, cfg.mirrored_input) if cfg.hands else None

    previews: dict[str, list[np.ndarray]] = {"body": [], "face": [], "hands": []}
    timings = {"body": 0.0, "face": 0.0, "hands": 0.0}
    n = 0
    try:
        for idx, ts, rgb in iter_frames(video, cfg.target_fps, cfg.max_frames):
            n += 1
            ts_ms = int(round(ts))
            bk = fk = hk = None
            if body is not None:
                t = time.perf_counter(); bk = body.process(rgb, ts_ms); timings["body"] += time.perf_counter() - t
            if face is not None:
                t = time.perf_counter(); fk = face.process(rgb, ts_ms); timings["face"] += time.perf_counter() - t
            if hands is not None:
                t = time.perf_counter()
                hk = hands.process(rgb, ts_ms, bk if body is not None and body.present[-1] else None)
                timings["hands"] += time.perf_counter() - t

            h, w = rgb.shape[:2]
            if cfg.write_previews:
                if cfg.body and bk is not None:
                    layer = R.draw_mp_body(R.blank(h, w), R.body_xyv(bk))
                    previews["body"].append(R.put_label(R.overlay(rgb, layer), f"body {idx}"))
                if face is not None:
                    layer = R.draw_face_points(R.blank(h, w), fk) if fk is not None else R.blank(h, w)
                    previews["face"].append(R.put_label(R.overlay(rgb, layer, 0.8), f"face {idx}"))
                if hands is not None:
                    layer = R.blank(h, w)
                    for s in range(2):
                        if hands.present[-1][s]:
                            R.draw_openpose_hand(layer, hk[s])
                    previews["hands"].append(R.put_label(R.overlay(rgb, layer, 0.8), f"hands {idx}"))
    finally:
        for ex in (body, face, hands):
            if ex is not None:
                ex.close()

    if n == 0:
        raise RuntimeError(f"No frames decoded from {video}")

    common_meta = {"format_version": FORMAT_VERSION, "source": video.name, "fps": fps,
                   "width": meta.width, "height": meta.height, "num_frames": n}
    summary: dict[str, Any] = {**common_meta, "timings_s": {k: round(v, 2) for k, v in timings.items()}}

    if cfg.smoothing not in ("none", "one_euro"):
        raise ValueError(f"smoothing must be 'none' or 'one_euro', got {cfg.smoothing!r}")
    b = body.result() if body is not None else None
    hd = f = None
    if cfg.body and body is not None:
        feats = _body_features(b, fps, cfg.smooth_alpha)
        if cfg.smoothing == "one_euro" and len(b["kp2d"]):
            feats["kp2d_smoothed"] = _smooth(b["kp2d"][..., :2], b["present"], fps, cfg)
        torch.save({"meta": common_meta, **_to_torch(b), "features": _to_torch(feats)}, out_dir / "body_motion.pt")
        summary["body"] = {"detection_rate": float(b["present"].mean()),
                           "jitter_xy": F.temporal_jitter(b["kp2d"][..., :2], b["present"])}
    if face is not None:
        f = face.result()
        feats = _face_features(f, fps)
        if cfg.smoothing == "one_euro" and len(f["landmarks"]):
            feats["landmarks_smoothed"] = _smooth(f["landmarks"], f["present"], fps, cfg)
        torch.save({"meta": common_meta, **_to_torch(f), "features": _to_torch(feats)}, out_dir / "face_motion.pt")
        summary["face"] = {"detection_rate": float(f["present"].mean())}
    if hands is not None:
        hd = hands.result()
        if cfg.hand_postprocess:
            hd, post = postprocess_hands(hd, b["kp2d"] if b is not None and len(b["kp2d"]) else None,
                                         meta.width, meta.height, max_gap=cfg.hand_max_gap)
            summary["hands_postprocess"] = post
        feats = _hand_features(hd, fps)
        if cfg.smoothing == "one_euro" and len(hd["kp2d"]):
            feats["kp2d_smoothed"] = _smooth(hd.get("kp2d_filled", hd["kp2d"]),
                                             hd["present"] | hd.get("filled", np.zeros_like(hd["present"])), fps, cfg)
        torch.save({"meta": common_meta, **_to_torch(hd), "features": _to_torch(feats)}, out_dir / "hand_motion.pt")
        summary["hands"] = {"detection_rate_left": float(hd["present"][:, 0].mean()),
                            "detection_rate_right": float(hd["present"][:, 1].mean())}

    for key, name in (("body", "body_preview.mp4"), ("face", "face_preview.mp4"), ("hands", "hands_preview.mp4")):
        if previews[key]:
            write_video(out_dir / name, previews[key], fps)
    if cfg.write_openpose:
        write_video(out_dir / "pose_openpose.mp4", _control_frames(b, hd, f, meta.width, meta.height, fps, cfg), fps)
        summary["control"] = {"smoothing": cfg.smoothing, "hand_postprocess": cfg.hand_postprocess}

    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    log.info("Extraction done: %s", json.dumps(summary))
    return summary
