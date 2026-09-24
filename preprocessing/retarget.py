"""Bone-length retargeting: keep the DRIVER's motion (bone directions, timing, root path) and impose the
REFERENCE character's proportions (limb lengths, head size, torso), so an adult's dance drives a child, a
long-legged figure or a stylised character without stretching them into the actor's body.

Method (per frame, parent before child along a kinematic tree rooted at the pelvis):

    child' = parent' + (child - parent) * s_bone        s_bone = L_reference / L_driver

- 3D world coordinates (metres): L are the reference's world bone lengths and the driver's median world bone
  lengths, so the output's world bone lengths equal the reference's exactly.
- 2D image coordinates: the same s_bone multiplies the driver's projected bone vector, so foreshortening (a limb
  pointing at the camera looks short) is preserved instead of being "corrected".
- Placement (`anchor="reference"`): the retargeted skeleton is expressed in the reference image's pixel frame:
  its first-frame pelvis lands on the reference pelvis and a global scale maps the first-frame torso onto the
  reference torso. Root displacement is scaled with the body, i.e. measured in torso lengths.
- Missing data is never invented: a joint is valid only if it and its whole parent chain are valid in that frame.
  Bones not visible in the reference use the mirrored bone, else the driver length times the torso ratio.

Hands follow the retargeted wrists (translated, scaled like the forearm); face landmarks follow the nose and
are scaled like the head. Pure geometry, no model: tested with synthetic skeletons (tests/test_retarget.py).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

PELVIS, NECK = 33, 34            # virtual joints appended to the 33 MediaPipe landmarks
N_EXT = 35

# (parent, child) in traversal order (parents are always placed before their children).
BONES: tuple[tuple[int, int], ...] = (
    (PELVIS, 23), (PELVIS, 24), (PELVIS, NECK),
    (NECK, 11), (NECK, 12), (NECK, 0),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (0, 9), (0, 10),
)
_MIRROR = {11: 12, 13: 14, 15: 16, 17: 18, 19: 20, 21: 22, 23: 24, 25: 26, 27: 28, 29: 30, 31: 32,
           1: 4, 2: 5, 3: 6, 7: 8, 9: 10}
_MIRROR.update({v: k for k, v in list(_MIRROR.items())})
TORSO_BONE = BONES.index((PELVIS, NECK))
HEAD_BONE = BONES.index((NECK, 0))
FOREARM = {0: BONES.index((13, 15)), 1: BONES.index((14, 16))}  # hand side (0 left, 1 right) -> forearm bone
WRIST = {0: 15, 1: 16}


def _mirror_bone(b: tuple[int, int]) -> tuple[int, int]:
    return (_MIRROR.get(b[0], b[0]), _MIRROR.get(b[1], b[1]))


def extend(points: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(..., 33, D) + (..., 33) -> (..., 35, D) with pelvis = hip midpoint and neck = shoulder midpoint."""
    pelvis = (points[..., 23, :] + points[..., 24, :]) / 2
    neck = (points[..., 11, :] + points[..., 12, :]) / 2
    ext = np.concatenate([points, pelvis[..., None, :], neck[..., None, :]], axis=-2)
    v = np.concatenate([valid, (valid[..., 23] & valid[..., 24])[..., None], (valid[..., 11] & valid[..., 12])[..., None]],
                       axis=-1)
    return ext, v


def bone_lengths(points: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame lengths (..., n_bones) and validity for extended points (..., 35, D)."""
    p = np.array([b[0] for b in BONES])
    c = np.array([b[1] for b in BONES])
    lengths = np.linalg.norm(points[..., c, :] - points[..., p, :], axis=-1)
    return lengths, valid[..., p] & valid[..., c] & (lengths > 1e-9)


@dataclass
class Proportions:
    """Bone lengths of one body. `lengths[i]` is None when bone i was not measurable."""

    lengths: list[float | None]
    source: str                              # "world" (metres) or "image" (pixels)
    pelvis_px: list[float] | None = None     # reference placement (pixels) for anchor="reference"
    torso_px: float | None = None
    image_size: list[int] | None = None      # [width, height] of the reference image
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def filled(self, fallback: np.ndarray, torso_ratio: float) -> np.ndarray:
        """Missing bones: mirrored bone, else fallback (driver) length times the torso ratio."""
        out = np.empty(len(BONES))
        for i, b in enumerate(BONES):
            if self.lengths[i] is not None:
                out[i] = self.lengths[i]
                continue
            j = BONES.index(_mirror_bone(b)) if _mirror_bone(b) in BONES else None
            out[i] = self.lengths[j] if j is not None and self.lengths[j] is not None else fallback[i] * torso_ratio
        return out


def reference_proportions(kp2d: np.ndarray, kp3d_world: np.ndarray | None, width: int, height: int,
                          visibility: float = 0.5) -> Proportions:
    """Proportions of the reference character from ONE frame of MediaPipe body landmarks."""
    kp2d = np.asarray(kp2d, dtype=np.float64)
    valid = np.isfinite(kp2d).all(-1) & (kp2d[:, 3] >= visibility)
    px = kp2d[:, :2] * np.array([width, height])
    ext2, v2 = extend(px, valid)
    notes = []
    if kp3d_world is not None and np.isfinite(kp3d_world).all():
        ext, v = extend(np.asarray(kp3d_world, dtype=np.float64), valid)
        source = "world"
    else:
        ext, v = ext2, v2
        source = "image"
        notes.append("no world landmarks: image-plane lengths (foreshortened limbs in the reference shrink)")
    lengths, lv = bone_lengths(ext, v)
    if not lv[TORSO_BONE]:
        raise ValueError("Reference torso (both shoulders and both hips) is not visible; cannot retarget")
    missing = [f"{a}-{b}" for (a, b), ok in zip(BONES, lv) if not ok]
    if missing:
        notes.append(f"{len(missing)} bones not visible in the reference; filled from mirror/driver: {missing}")
    return Proportions([float(x) if ok else None for x, ok in zip(lengths, lv)], source,
                       pelvis_px=ext2[PELVIS].tolist() if v2[PELVIS] else None,
                       torso_px=float(np.linalg.norm(ext2[NECK] - ext2[PELVIS])) if v2[TORSO_BONE] else None,
                       image_size=[int(width), int(height)], notes=notes)


def driver_lengths(points: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Robust per-bone driver lengths: median over frames where the bone is valid (NaN if never valid)."""
    lengths, lv = bone_lengths(points, valid)
    out = np.full(len(BONES), np.nan)
    for i in range(len(BONES)):
        if lv[:, i].any():
            out[i] = np.median(lengths[lv[:, i], i])
    return out


def _chain(points: np.ndarray, valid: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply per-bone scale along the tree for (T, 35, D); root stays in place."""
    out = np.zeros_like(points)
    ok = np.zeros_like(valid)
    out[:, PELVIS] = points[:, PELVIS]
    ok[:, PELVIS] = valid[:, PELVIS]
    for i, (p, c) in enumerate(BONES):
        vec = points[:, c] - points[:, p]
        out[:, c] = out[:, p] + vec * scale[i]
        ok[:, c] = ok[:, p] & valid[:, c]
    return out, ok


def retarget_body(body: dict[str, Any], target: Proportions, *, visibility: float = 0.5,
                  anchor: str = "reference") -> tuple[dict[str, Any], dict[str, Any]]:
    """Retarget a body track (as saved in body_motion.pt). Returns (new track, report)."""
    meta = dict(body["meta"])
    w, h = meta["width"], meta["height"]
    kp2d = np.asarray(body["kp2d"], dtype=np.float64)
    present = np.asarray(body["present"], bool)
    world = np.asarray(body["kp3d_world"], dtype=np.float64) if "kp3d_world" in body else None
    valid = present[:, None] & np.isfinite(kp2d).all(-1) & (kp2d[..., 3] >= visibility)

    px, pv = extend(kp2d[..., :2] * np.array([w, h]), valid)
    if target.source == "world":
        if world is None:
            raise ValueError("Reference proportions are metric (world) but the driver track has no kp3d_world")
        wext, wv = extend(world, valid)
        drv = driver_lengths(wext, wv)
    else:
        wext = wv = None
        drv = driver_lengths(px, pv)
    if not np.isfinite(drv[TORSO_BONE]):
        raise ValueError("Driver torso is never visible; cannot retarget")
    torso_ratio = target.filled(drv, 1.0)[TORSO_BONE] / drv[TORSO_BONE]
    tgt = target.filled(np.where(np.isfinite(drv), drv, 0.0), torso_ratio)
    scale = np.where(np.isfinite(drv) & (drv > 0), tgt / np.where(drv > 0, drv, 1), torso_ratio)

    out2, ok2 = _chain(px, pv, scale)
    # Root path scaled with the body: a smaller character covers proportionally less ground (torso lengths kept).
    first_root = np.flatnonzero(pv[:, PELVIS])
    if first_root.size:
        p0 = px[first_root[0], PELVIS]
        out2 += ((px[:, PELVIS] - p0) * (torso_ratio - 1.0))[:, None, :]
    report: dict[str, Any] = {"source": target.source, "bone_scale": scale.round(4).tolist(),
                              "torso_ratio": float(torso_ratio), "anchor": anchor, "notes": list(target.notes)}
    ow, oh = w, h
    if anchor == "reference":
        if target.pelvis_px is None or target.torso_px is None or target.image_size is None:
            raise ValueError("anchor='reference' needs the reference pelvis/torso in image space")
        first = np.flatnonzero(ok2[:, PELVIS] & ok2[:, NECK])
        if not first.size:
            raise ValueError("No driver frame with a visible torso to anchor on")
        f0 = first[0]
        g = target.torso_px / np.linalg.norm(out2[f0, NECK] - out2[f0, PELVIS])
        out2 = np.asarray(target.pelvis_px) + (out2 - out2[f0, PELVIS]) * g
        ow, oh = target.image_size
        report.update(global_scale=float(g), anchor_frame=int(f0))
    elif anchor != "driver":
        raise ValueError("anchor must be 'reference' or 'driver'")
    else:
        g = 1.0

    new = {k: v for k, v in body.items()}
    k2 = np.array(kp2d, dtype=np.float32, copy=True)
    k2[..., 0] = out2[:, :33, 0] / ow
    k2[..., 1] = out2[:, :33, 1] / oh
    k2[..., 2] = kp2d[..., 2] * g * (w / ow)            # relative depth, same scale as x
    k2[..., 3] = np.where(ok2[:, :33], kp2d[..., 3], 0.0)
    new["kp2d"] = k2
    if wext is not None:
        wout, _ = _chain(wext, wv, scale)
        new["kp3d_world"] = (wout[:, :33] - wout[:, PELVIS:PELVIS + 1]).astype(np.float32)  # hip-centred as MediaPipe
    new["present"] = present & ok2[:, PELVIS]
    new.pop("features", None)  # derived features no longer match; recompute from the new track if needed
    meta.update(width=int(ow), height=int(oh))
    new["meta"] = meta
    report["joint_scale_px"] = float(g)
    report["_placed"] = (out2, ok2, scale, g, (w, h), (ow, oh))
    return new, report


def _follow(points_px: np.ndarray, old_anchor: np.ndarray, new_anchor: np.ndarray, factor: float) -> np.ndarray:
    return new_anchor[:, None, :] + (points_px - old_anchor[:, None, :]) * factor


def retarget_hands(hands: dict[str, Any], placed) -> dict[str, Any]:
    out2, ok2, scale, g, (w, h), (ow, oh) = placed
    new = {k: v for k, v in hands.items()}
    kp = np.array(hands["kp2d"], dtype=np.float64, copy=True)
    present = np.array(hands["present"], bool, copy=True)
    for side in (0, 1):
        wr = WRIST[side]
        f = scale[FOREARM[side]] * g
        pts = kp[:, side, :, :2] * np.array([w, h])
        moved = _follow(pts, pts[:, 0], out2[:, wr], f)  # hand wrist lands on the retargeted body wrist
        kp[:, side, :, 0], kp[:, side, :, 1] = moved[..., 0] / ow, moved[..., 1] / oh
        kp[:, side, :, 2] *= f * (w / ow)
        present[:, side] &= ok2[:, wr]
    new["kp2d"], new["present"] = kp.astype(np.float32), present
    new.pop("features", None)
    new["meta"] = {**hands["meta"], "width": int(ow), "height": int(oh)}
    return new


def retarget_face(face: dict[str, Any], body_old_px: np.ndarray, placed) -> dict[str, Any]:
    out2, ok2, scale, g, (w, h), (ow, oh) = placed
    new = {k: v for k, v in face.items()}
    lm = np.array(face["landmarks"], dtype=np.float64, copy=True)
    f = scale[HEAD_BONE] * g
    pts = lm[..., :2] * np.array([w, h])
    moved = _follow(pts, body_old_px[:, 0], out2[:, 0], f)  # follow the retargeted nose
    lm[..., 0], lm[..., 1] = moved[..., 0] / ow, moved[..., 1] / oh
    lm[..., 2] *= f * (w / ow)
    new["landmarks"] = lm.astype(np.float32)
    new["present"] = np.asarray(face["present"], bool) & ok2[:, 0]
    new.pop("features", None)
    new["meta"] = {**face["meta"], "width": int(ow), "height": int(oh)}
    return new


def retarget_tracks(tracks: dict[str, dict[str, Any]], target: Proportions, *, visibility: float = 0.5,
                    anchor: str = "reference") -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Retarget {"body", "face"?, "hands"?} tracks. Face/hands follow the body; returns (tracks, report)."""
    body = tracks["body"]
    new_body, report = retarget_body(body, target, visibility=visibility, anchor=anchor)
    placed = report.pop("_placed")
    w, h = body["meta"]["width"], body["meta"]["height"]
    old_px = np.asarray(body["kp2d"], dtype=np.float64)[..., :2] * np.array([w, h])
    out = {"body": new_body}
    if tracks.get("hands") is not None:
        out["hands"] = retarget_hands(tracks["hands"], placed)
    if tracks.get("face") is not None:
        out["face"] = retarget_face(tracks["face"], old_px, placed)
    for t in out.values():
        t["meta"]["retarget"] = {k: v for k, v in report.items() if k != "bone_scale"}
    report["target"] = {k: v for k, v in target.to_dict().items() if k != "lengths"}
    return out, report


def retarget_directory(motion_dir, reference_rgb: np.ndarray, out_dir, *, visibility: float = 0.5,
                       anchor: str = "reference", write_control: bool = True) -> dict[str, Any]:
    """Retarget the .pt tracks of `motion_dir` to the body in `reference_rgb`; writes the same file names plus
    pose_openpose.mp4 rendered in the reference image frame. Raises if the reference torso is not visible."""
    import json
    from pathlib import Path

    import torch

    from common.video_io import write_video
    from evaluation.identity import analyze_reference

    from .control import render_control_frames

    motion_dir, out_dir = Path(motion_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks = {k: torch.load(motion_dir / f, map_location="cpu", weights_only=True)
              for k, f in (("body", "body_motion.pt"), ("face", "face_motion.pt"), ("hands", "hand_motion.pt"))
              if (motion_dir / f).is_file()}
    tracks = {k: {kk: (vv.numpy() if hasattr(vv, "numpy") else vv) for kk, vv in t.items()} for k, t in tracks.items()}
    ref_face, ref_body = analyze_reference(reference_rgb)
    if ref_body is None:
        raise ValueError("No body detected in the reference image; cannot retarget")
    world = _reference_world(reference_rgb)
    h, w = reference_rgb.shape[:2]
    target = reference_proportions(ref_body, world, w, h, visibility)
    new, report = retarget_tracks(tracks, target, visibility=visibility, anchor=anchor)
    names = {"body": "body_motion.pt", "face": "face_motion.pt", "hands": "hand_motion.pt"}
    for k, t in new.items():
        torch.save({kk: (torch.from_numpy(np.ascontiguousarray(vv)) if isinstance(vv, np.ndarray) else vv)
                    for kk, vv in t.items()}, out_dir / names[k])
    if write_control:
        meta = new["body"]["meta"]
        frames = render_control_frames(new["body"], new.get("hands"), new.get("face"), meta["width"], meta["height"])
        write_video(out_dir / "pose_openpose.mp4", frames, meta["fps"])
    (out_dir / "retarget.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _reference_world(rgb: np.ndarray) -> np.ndarray | None:
    from preprocessing.pose.extract_body import BodyExtractor

    ex = BodyExtractor("pose_landmarker_full")
    try:
        ex.process(np.ascontiguousarray(rgb), 0)
        r = ex.result()
    finally:
        ex.close()
    return r["kp3d_world"][0] if len(r["present"]) and r["present"][0] else None
