"""Detector-based motion diagnostics. No interpolation or quality PASS claims."""

from pathlib import Path

import numpy as np
import torch

METRIC_VERSION = "motion-v1"


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
    result["identity"] = {"status": "NOT_MEASURED"}
    result["visual_quality"] = {"status": "REQUIRES_HUMAN_REVIEW"}
    return result


def load_tracks(directory):
    # Only tensor/primitives payloads are accepted; no arbitrary pickle execution.
    return {kind: torch.load(Path(directory) / filename, map_location="cpu", weights_only=True)
            for kind, filename in (("body", "body_motion.pt"), ("face", "face_motion.pt"),
                                   ("hands", "hand_motion.pt"))}
