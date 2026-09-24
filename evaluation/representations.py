"""EXP-003 (partial, synthetic): which body-motion representation should the Motion Encoder consume?

A procedural 3D rig (same bone tree as preprocessing/retarget.py) performs several motions, is filmed by a
perspective camera from several viewpoints/zooms, with an adult and a child body, then corrupted by Gaussian keypoint
noise (2D pixels and 3D world). Dropouts are NOT modelled yet. Each candidate representation is scored on:

- nuisance_ratio: distance between two renderings of the SAME motion that differ only in camera or body size,
  divided by the motion's own variation. Lower = the representation ignores what should not matter.
- noise_jitter_ratio: jitter (|2nd difference|) added by keypoint noise, divided by the clean motion's own
  frame-to-frame change. Lower = more robust to detector noise.
- pca_components_95: components needed for 95 % of the variance across all motions (compactness).
- prediction_error: constant-velocity one-step prediction error / mean step (smoothness/predictability).

Everything is synthetic: the noise model (Gaussian pixels + relative 3D noise) is an assumption, not a
measurement of MediaPipe. Numbers rank representations under these assumptions; EXP-003 on real videos is pending.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from preprocessing.features import normalize_body
from preprocessing.retarget import BONES, NECK, PELVIS, extend

REST = np.array([
    (0, .72, .08), (.03, .76, .07), (.045, .76, .07), (.06, .76, .06), (-.03, .76, .07), (-.045, .76, .07),
    (-.06, .76, .06), (.08, .74, 0), (-.08, .74, 0), (.03, .68, .07), (-.03, .68, .07), (.18, .5, 0), (-.18, .5, 0),
    (.2, .22, 0), (-.2, .22, 0), (.22, -.05, 0), (-.22, -.05, 0), (.23, -.12, .01), (-.23, -.12, .01),
    (.22, -.13, 0), (-.22, -.13, 0), (.24, -.1, .02), (-.24, -.1, .02), (.1, 0, 0), (-.1, 0, 0), (.1, -.42, .02),
    (-.1, -.42, .02), (.1, -.82, 0), (-.1, -.82, 0), (.1, -.86, -.04), (-.1, -.86, -.04), (.1, -.86, .1),
    (-.1, -.86, .1)], float)
SUBTREES = {"l_arm": (13, 15, 17, 19, 21), "r_arm": (14, 16, 18, 20, 22), "l_leg": (25, 27, 29, 31),
            "r_leg": (26, 28, 30, 32), "l_fore": (15, 17, 19, 21), "r_fore": (16, 18, 20, 22)}
PIVOT = {"l_arm": 11, "r_arm": 12, "l_leg": 23, "r_leg": 24, "l_fore": 13, "r_fore": 14}
CHILD = {(PELVIS, NECK): 0.75, (23, 25): 0.6, (25, 27): 0.6, (24, 26): 0.6, (26, 28): 0.6, (11, 13): 0.7,
         (13, 15): 0.7, (12, 14): 0.7, (14, 16): 0.7, (NECK, 0): 1.2}


def _rx(a):
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def _ry(a):
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def _rz(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def scaled_rest(scales: dict | None = None) -> np.ndarray:
    ext, _ = extend(REST[None], np.ones((1, 33), bool))
    out = ext.copy()
    for p, c in BONES:
        out[:, c] = out[:, p] + (ext[:, c] - ext[:, p]) * (scales or {}).get((p, c), 1.0)
    return out[0, :33]


MOTIONS: dict[str, Callable[[float], dict[str, np.ndarray]]] = {
    "walk": lambda t: {"l_arm": _rx(0.5 * np.sin(2 * np.pi * t)), "r_arm": _rx(-0.5 * np.sin(2 * np.pi * t)),
                       "l_leg": _rx(-0.45 * np.sin(2 * np.pi * t)), "r_leg": _rx(0.45 * np.sin(2 * np.pi * t))},
    "dance": lambda t: {"l_arm": _rz(1.2 + 0.8 * np.sin(4 * np.pi * t)), "r_arm": _rz(-1.2 - 0.8 * np.sin(4 * np.pi * t + 1)),
                        "l_fore": _rz(0.9 * np.sin(6 * np.pi * t)), "r_fore": _rz(-0.9 * np.sin(6 * np.pi * t)),
                        "l_leg": _rz(0.25 * np.sin(4 * np.pi * t))},
    "turn": lambda t: {"root": _ry(1.4 * np.sin(np.pi * t)), "l_arm": _rz(0.6), "r_arm": _rz(-0.6)},
    "kick": lambda t: {"r_leg": _rx(-1.3 * max(0.0, np.sin(2 * np.pi * t))), "l_arm": _rz(0.8), "r_arm": _rz(-0.8)},
}


def animate(motion: str, rest: np.ndarray, frames: int = 48, fps: float = 16.0, travel: float = 0.6) -> np.ndarray:
    """World joints (T, 33, 3), hip-centred per frame plus a root path added separately (returned in [..., :])."""
    out = np.repeat(rest[None], frames, 0).astype(float)
    for i in range(frames):
        pose = MOTIONS[motion](i / fps)
        for name in ("l_arm", "r_arm", "l_leg", "r_leg"):          # parents first
            if name in pose:
                idx = list(SUBTREES[name])
                out[i, idx] = (out[i, idx] - out[i, PIVOT[name]]) @ pose[name].T + out[i, PIVOT[name]]
        for name in ("l_fore", "r_fore"):
            if name in pose:
                idx = list(SUBTREES[name])
                out[i, idx] = (out[i, idx] - out[i, PIVOT[name]]) @ pose[name].T + out[i, PIVOT[name]]
        if "root" in pose:
            out[i] = out[i] @ pose["root"].T
    root = np.zeros((frames, 3))
    if motion == "walk":
        root[:, 0] = np.linspace(0, travel, frames)
    return out, root


def project(world: np.ndarray, root: np.ndarray, *, yaw: float = 0.0, distance: float = 3.5, focal: float = 1.2,
            shift=(0.0, 0.0)) -> np.ndarray:
    """Perspective camera looking at the pelvis; returns normalised image coords (T, 33, 2) in a square image."""
    p = (world + root[:, None]) @ _ry(yaw).T
    z = distance - p[..., 2]
    x = focal * p[..., 0] / z + 0.5 + shift[0]
    y = -focal * p[..., 1] / z + 0.5 + shift[1]
    return np.stack([x, y], -1)


# Long bones only: torso, shoulders/hips, upper/lower arms and legs (no face points, fingers or feet).
LIMB_BONES = ((PELVIS, 23), (PELVIS, 24), (PELVIS, NECK), (NECK, 11), (NECK, 12), (NECK, 0), (11, 13), (13, 15),
              (12, 14), (14, 16), (23, 25), (25, 27), (24, 26), (26, 28))


def bone_rot6d(world: np.ndarray, rest: np.ndarray, bones=BONES) -> np.ndarray:
    """Per-bone swing rotation (rest direction -> current direction) in 6D, (T, n_bones * 6). Twist is unobservable
    from joint positions and is omitted."""
    ext, _ = extend(world, np.ones(world.shape[:2], bool))
    rext, _ = extend(rest[None], np.ones((1, 33), bool))
    feats = []
    for p, c in bones:
        a = rext[0, c] - rext[0, p]
        a = a / np.linalg.norm(a)
        b = ext[:, c] - ext[:, p]
        b = b / np.linalg.norm(b, axis=-1, keepdims=True)
        v = np.cross(a[None], b)
        s = np.linalg.norm(v, axis=-1, keepdims=True)
        cth = (b @ a)[:, None]
        k = np.where(s > 1e-9, v / np.maximum(s, 1e-12), 0.0)
        kx = np.zeros((len(b), 3, 3))
        kx[:, 0, 1], kx[:, 0, 2], kx[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
        kx = kx - kx.transpose(0, 2, 1)
        r = np.eye(3)[None] + s[..., None] * kx + (1 - cth)[..., None] * (kx @ kx)
        feats.append(r[:, :, :2].reshape(len(b), 6))
    return np.concatenate(feats, -1)


def representations(image_xy: np.ndarray, world: np.ndarray, rest: np.ndarray) -> dict[str, np.ndarray]:
    n2d, _, _ = normalize_body(image_xy.astype(np.float32))
    ext, _ = extend(world, np.ones(world.shape[:2], bool))
    torso = np.linalg.norm(ext[:, NECK] - ext[:, PELVIS], axis=-1)
    return {
        "2d_image": image_xy.reshape(len(image_xy), -1),
        "2d_normalized": n2d.reshape(len(n2d), -1),
        "3d_world": world.reshape(len(world), -1),
        "3d_world_normalized": (world / torso[:, None, None]).reshape(len(world), -1),
        "rot6d_bones": bone_rot6d(world, rest),
        "rot6d_limbs": bone_rot6d(world, rest, LIMB_BONES),
        "3d_normalized+rot6d_limbs": np.concatenate([(world / torso[:, None, None]).reshape(len(world), -1),
                                                     bone_rot6d(world, rest, LIMB_BONES)], -1),
    }


def _energy(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(x, axis=0), axis=-1).mean()) + 1e-12


def _jitter(x: np.ndarray) -> float:
    return float(np.linalg.norm(x[2:] - 2 * x[1:-1] + x[:-2], axis=-1).mean())


def run(seed: int = 0, frames: int = 48, pixel_noise: float = 0.004, world_noise: float = 0.01) -> dict:
    rng = np.random.default_rng(seed)
    adult, child = REST.copy(), scaled_rest(CHILD)
    cams = {"front": dict(), "yaw30": dict(yaw=0.5), "zoom": dict(distance=2.5), "shift": dict(shift=(0.15, -0.05))}
    names = list(representations(np.zeros((3, 33, 2)), np.repeat(REST[None], 3, 0), REST))
    nuisance = {n: {"camera": [], "body_size": []} for n in names}
    noise = {n: [] for n in names}
    pred = {n: [] for n in names}
    pooled = {n: [] for n in names}
    for motion in MOTIONS:
        w_adult, root = animate(motion, adult, frames)
        w_child, _ = animate(motion, child, frames)
        base = representations(project(w_adult, root), w_adult, adult)
        for n in names:
            e = _energy(base[n])
            pooled[n].append(base[n])
            for cam, kw in cams.items():
                if cam == "front":
                    continue
                other = representations(project(w_adult, root, **kw), w_adult, adult)[n]
                nuisance[n]["camera"].append(np.linalg.norm(other - base[n], axis=-1).mean() / e)
            kid = representations(project(w_child, root * 0.75), w_child, child)[n]
            nuisance[n]["body_size"].append(np.linalg.norm(kid - base[n], axis=-1).mean() / e)
            steps = np.linalg.norm(np.diff(base[n], axis=0), axis=-1).mean() + 1e-12
            cv = 2 * base[n][1:-1] - base[n][:-2]
            pred[n].append(np.linalg.norm(cv - base[n][2:], axis=-1).mean() / steps)
        img = project(w_adult, root) + rng.normal(0, pixel_noise, (frames, 33, 2))
        scale = np.linalg.norm(adult[11] - adult[23])
        wn = w_adult + rng.normal(0, world_noise * scale, w_adult.shape)
        noisy = representations(img, wn, adult)
        for n in names:
            noise[n].append((_jitter(noisy[n]) - _jitter(base[n])) / _energy(base[n]))
    out = {}
    for n in names:
        allx = np.concatenate(pooled[n])
        sv = np.linalg.svd(allx - allx.mean(0), compute_uv=False) ** 2
        comps = int(np.searchsorted(np.cumsum(sv) / sv.sum(), 0.95) + 1)
        out[n] = {"nuisance_camera": float(np.mean(nuisance[n]["camera"])),
                  "nuisance_body_size": float(np.mean(nuisance[n]["body_size"])),
                  "noise_jitter_ratio": float(np.mean(noise[n])), "pca_components_95": comps,
                  "dims": int(allx.shape[1]), "prediction_error": float(np.mean(pred[n]))}
    return {"experiment": "EXP-003-synthetic", "seed": seed, "frames": frames, "motions": list(MOTIONS),
            "cameras": list(cams), "pixel_noise": pixel_noise, "world_noise_rel_torso": world_noise,
            "results": out}
# Caveat for 3D rows: the synthetic "world" coordinates are the ground-truth rig, so their camera invariance is 0 by
# construction. MediaPipe world landmarks are estimated per image and are not perfectly camera invariant.
