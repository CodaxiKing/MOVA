"""Detector-based motion diagnostics. No interpolation or quality PASS claims."""

from pathlib import Path

import numpy as np
import torch

METRIC_VERSION = "motion-v2"


def _array(track, key, shape):
    value = np.asarray(track[key], dtype=np.float64)
    if value.shape != shape:
        raise ValueError(f"{key}: expected {shape}, got {value.shape}")
    return value


def _mean(values):
    return float(np.mean(values)) if values.size else None


def _normalized(track, kind, n, visibility):
    meta = track["meta"]
    size = np.array([meta["width"], meta["height"]], dtype=float)
    if not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError("Invalid track image dimensions")
    if kind == "body":
        kp = _array(track, "kp2d", (n, 33, 4))
        present = _array(track, "present", (n,)) == 1
        valid = present[:, None] & np.isfinite(kp).all(-1) & (kp[..., 3] >= visibility)
        xy = kp[..., :2] * size
        center = (xy[:, 23] + xy[:, 24]) / 2
        anchor = (xy[:, 11] + xy[:, 12]) / 2
        scale = np.linalg.norm(anchor - center, axis=-1)
        usable = valid[:, [11, 12, 23, 24]].all(-1) & (scale > 1e-6)
        valid &= usable[:, None]
        xy = (xy - center[:, None]) / np.where(usable, scale, 1)[:, None, None]
    else:
        kp = _array(track, "kp2d", (n, 2, 21, 3))
        present = _array(track, "present", (n, 2)) == 1
        valid = present[..., None] & np.isfinite(kp).all(-1)
        xy = kp[..., :2] * size
        scale = np.linalg.norm(xy[:, :, 9] - xy[:, :, 0], axis=-1)
        usable = valid[:, :, [0, 9]].all(-1) & (scale > 1e-6)
        valid &= usable[..., None]
        xy = (xy - xy[:, :, :1]) / np.where(usable, scale, 1)[..., None, None]
        xy, valid = xy.reshape(n, 42, 2), valid.reshape(n, 42)
    return xy, valid


def keypoint_metrics(reference, generated, ref_valid, gen_valid, *, fps, threshold):
    paired = ref_valid & gen_valid
    distances = np.linalg.norm(generated - reference, axis=-1)
    ref_count, pair_count = int(ref_valid.sum()), int(paired.sum())
    # Missing generated detections count as incorrect, never as perfect matches.
    correct = int((paired & (distances <= threshold)).sum())
    triples = paired[2:] & paired[1:-1] & paired[:-2]
    delta_ref = np.diff(reference, n=2, axis=0) * fps**2
    delta_gen = np.diff(generated, n=2, axis=0) * fps**2
    return {
        "reference_points": ref_count, "paired_points": pair_count,
        "reference_coverage": float(ref_valid.mean()),
        "generated_coverage": float(gen_valid.mean()),
        "paired_coverage": pair_count / ref_count if ref_count else None,
        "pck": correct / ref_count if ref_count else None,
        "pck_threshold": threshold,
        "mean_error_paired": _mean(distances[paired]),
        "temporal_triples": int(triples.sum()),
        "acceleration_error_paired": _mean(np.linalg.norm(delta_gen - delta_ref, axis=-1)[triples]),
        "reference_acceleration_paired": _mean(np.linalg.norm(delta_ref, axis=-1)[triples]),
        "generated_acceleration_paired": _mean(np.linalg.norm(delta_gen, axis=-1)[triples]),
    }


def compare_tracks(reference, generated, *, pck_threshold=0.1, visibility=0.5):
    if not np.isfinite(pck_threshold) or pck_threshold <= 0 or not 0 <= visibility <= 1:
        raise ValueError("Invalid metric thresholds")
    meta = reference["body"]["meta"]
    n, fps = meta["num_frames"], meta["fps"]
    if not isinstance(n, int) or n < 1 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid frame count or FPS")
    for tracks in (reference, generated):
        for kind in ("body", "hands", "face"):
            m = tracks[kind]["meta"]
            if m["format_version"] != 1 or m["num_frames"] != n or not np.isclose(m["fps"], fps, atol=0.01, rtol=0):
                raise ValueError("Tracks must have matching frame count/FPS and format version 1")
    result = {"metric_version": METRIC_VERSION, "frames": n, "fps": fps}
    for kind in ("body", "hands"):
        ref, rv = _normalized(reference[kind], kind, n, visibility)
        gen, gv = _normalized(generated[kind], kind, n, visibility)
        result[kind] = keypoint_metrics(ref, gen, rv, gv, fps=fps, threshold=pck_threshold)
        if kind == "hands":
            for side, indices in (("left", slice(0, 21)), ("right", slice(21, 42))):
                result[kind][side] = keypoint_metrics(ref[:, indices], gen[:, indices], rv[:, indices],
                                                     gv[:, indices], fps=fps, threshold=pck_threshold)
    r, g = reference["face"], generated["face"]
    rb, gb = _array(r, "blendshapes", (n, 52)), _array(g, "blendshapes", (n, 52))
    rp = (_array(r, "present", (n,)) == 1) & np.isfinite(rb).all(-1)
    gp = (_array(g, "present", (n,)) == 1) & np.isfinite(gb).all(-1)
    paired = rp & gp
    if paired.any() and (len(r["blendshape_names"]) != 52 or r["blendshape_names"] != g["blendshape_names"]):
        raise ValueError("Face blendshape ordering mismatch")
    result["face"] = {
        "reference_coverage": float(rp.mean()), "generated_coverage": float(gp.mean()),
        "paired_frames": int(paired.sum()),
        "paired_coverage": float(paired.sum() / rp.sum()) if rp.any() else None,
        "blendshape_mae_paired": _mean(np.abs(rb[paired] - gb[paired])),
    }
    result["trajectory"] = trajectory_metrics(reference["body"], generated["body"], n, visibility)
    result["head_rotation"] = head_rotation_metrics(r, g, n)
    result["body_orientation"] = body_orientation_metrics(reference["body"], generated["body"], n, visibility)
    result["identity"] = {"status": "NOT_MEASURED"}
    result["visual_quality"] = {"status": "REQUIRES_HUMAN_REVIEW"}
    return result


# --- Global motion (motion-v2) ------------------------------------------------------------------------------
# Pose metrics above are root-centered and torso-scaled, so they cannot see where the body goes, how its
# apparent size changes, or how the head/torso rotate. These metrics measure exactly that.

def _unavailable(reason):
    return {"status": "UNAVAILABLE", "reason": reason}


def _root_and_torso(track, n, visibility):
    meta = track["meta"]
    size = np.array([meta["width"], meta["height"]], dtype=float)
    kp = _array(track, "kp2d", (n, 33, 4))
    present = _array(track, "present", (n,)) == 1
    anchors = kp[:, [11, 12, 23, 24]]
    valid = present & np.isfinite(anchors).all((-1, -2)) & (anchors[..., 3] >= visibility).all(-1)
    xy = kp[..., :2] * size
    root = (xy[:, 23] + xy[:, 24]) / 2
    torso = np.linalg.norm((xy[:, 11] + xy[:, 12]) / 2 - root, axis=-1)
    valid &= torso > 1e-6
    return root, torso, valid, float(np.hypot(*size))


def trajectory_metrics(reference, generated, n, visibility):
    """Root (hip-center) path and apparent scale, each relative to the first frame both tracks share.

    Displacements are divided by each track's own mean torso length, so a taller character following the
    same path scores the same. `absolute_root_error` keeps framing information (fraction of image diagonal).
    """
    rr, rt, rv, diag = _root_and_torso(reference, n, visibility)
    gr, gt, gv, gdiag = _root_and_torso(generated, n, visibility)
    paired = rv & gv
    if not paired.any():
        return {"paired_frames": 0, "paired_coverage": 0.0 if rv.any() else None, "trajectory_error": None,
                "final_displacement_error": None, "scale_log_error": None, "absolute_root_error": None,
                "reference_path_length": None}
    f0 = int(np.argmax(paired))
    rel_r = (rr - rr[f0]) / rt[rv].mean()
    rel_g = (gr - gr[f0]) / gt[gv].mean()
    err = np.linalg.norm(rel_g - rel_r, axis=-1)[paired]
    scale_err = np.abs(np.log(gt[paired] / gt[f0]) - np.log(rt[paired] / rt[f0]))
    abs_err = np.linalg.norm(gr[paired] / gdiag - rr[paired] / diag, axis=-1)
    idx = np.flatnonzero(paired)
    steps = np.linalg.norm(np.diff(rel_r[rv], axis=0), axis=-1)
    return {
        "paired_frames": int(paired.sum()),
        "paired_coverage": float(paired.sum() / rv.sum()),
        "trajectory_error": float(err.mean()),
        "final_displacement_error": float(np.linalg.norm(rel_g[idx[-1]] - rel_r[idx[-1]])),
        "scale_log_error": float(scale_err.mean()),
        "absolute_root_error": float(abs_err.mean()),
        "reference_path_length": float(steps.sum()) if steps.size else 0.0,
        "units": "torso lengths (trajectory), |log ratio| (scale), image diagonal (absolute)",
    }


def _nearest_rotation(m):
    u, _, vt = np.linalg.svd(m)
    d = np.sign(np.linalg.det(u @ vt))
    u[..., :, -1] *= d[..., None]
    return u @ vt


def _geodesic_deg(a, b):
    cos = (np.trace(np.swapaxes(a, -1, -2) @ b, axis1=-2, axis2=-1) - 1) / 2
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def head_rotation_metrics(reference, generated, n):
    """Geodesic angle between head rotations (MediaPipe facial transform), absolute and relative to frame 0."""
    if "head_transform" not in reference or "head_transform" not in generated:
        return _unavailable("head_transform missing from face tracks")
    ra = _array(reference, "head_transform", (n, 4, 4))[:, :3, :3]
    ga = _array(generated, "head_transform", (n, 4, 4))[:, :3, :3]
    rv = (_array(reference, "present", (n,)) == 1) & np.isfinite(ra).all((-1, -2))
    gv = (_array(generated, "present", (n,)) == 1) & np.isfinite(ga).all((-1, -2))
    paired = rv & gv
    if not paired.any():
        return {"paired_frames": 0, "paired_coverage": 0.0 if rv.any() else None, "geodesic_error_deg": None,
                "relative_geodesic_error_deg": None, "reference_rotation_range_deg": None}
    R, G = np.eye(3)[None].repeat(n, 0), np.eye(3)[None].repeat(n, 0)
    R[rv], G[gv] = _nearest_rotation(ra[rv]), _nearest_rotation(ga[gv])
    f0 = int(np.argmax(paired))
    rel_r = np.swapaxes(R[f0], -1, -2)[None] @ R
    rel_g = np.swapaxes(G[f0], -1, -2)[None] @ G
    return {
        "paired_frames": int(paired.sum()),
        "paired_coverage": float(paired.sum() / rv.sum()),
        "geodesic_error_deg": float(_geodesic_deg(R[paired], G[paired]).mean()),
        "relative_geodesic_error_deg": float(_geodesic_deg(rel_r[paired], rel_g[paired]).mean()),
        "reference_rotation_range_deg": float(_geodesic_deg(R[f0][None], R[rv]).max()),
    }


def _yaw_deg(track, n, visibility):
    kp = _array(track, "kp2d", (n, 33, 4))
    world = _array(track, "kp3d_world", (n, 33, 3))
    valid = ((_array(track, "present", (n,)) == 1) & (kp[:, [11, 12], 3] >= visibility).all(-1)
             & np.isfinite(world[:, [11, 12]]).all((-1, -2)))
    v = world[:, 11] - world[:, 12]  # right shoulder -> left shoulder
    valid &= np.hypot(v[:, 0], v[:, 2]) > 1e-6
    return np.degrees(np.arctan2(v[:, 2], v[:, 0])), valid


def _circular(d):
    return (d + 180.0) % 360.0 - 180.0


def body_orientation_metrics(reference, generated, n, visibility):
    """Torso yaw from 3D world shoulders; turning motions are invisible to root-centered 2D pose."""
    if "kp3d_world" not in reference or "kp3d_world" not in generated:
        return _unavailable("kp3d_world missing from body tracks")
    ry, rv = _yaw_deg(reference, n, visibility)
    gy, gv = _yaw_deg(generated, n, visibility)
    paired = rv & gv
    if not paired.any():
        return {"paired_frames": 0, "paired_coverage": 0.0 if rv.any() else None, "yaw_error_deg": None,
                "relative_yaw_error_deg": None, "reference_yaw_range_deg": None}
    f0 = int(np.argmax(paired))
    rel = _circular((gy - gy[f0]) - (ry - ry[f0]))
    return {
        "paired_frames": int(paired.sum()),
        "paired_coverage": float(paired.sum() / rv.sum()),
        "yaw_error_deg": float(np.abs(_circular(gy - ry))[paired].mean()),
        "relative_yaw_error_deg": float(np.abs(rel)[paired].mean()),
        "reference_yaw_range_deg": float(np.abs(_circular(ry[rv] - ry[f0])).max()),
    }


def load_tracks(directory):
    # Only tensor/primitives payloads are accepted; no arbitrary pickle execution.
    return {kind: torch.load(Path(directory) / filename, map_location="cpu", weights_only=True)
            for kind, filename in (("body", "body_motion.pt"), ("face", "face_motion.pt"),
                                   ("hands", "hand_motion.pt"))}
