"""Bone-length retargeting on synthetic animated skeletons (orthographic camera, exact invariants)."""

import numpy as np
import pytest

from preprocessing.control import render_control_frames
from preprocessing.retarget import (BONES, NECK, PELVIS, TORSO_BONE, bone_lengths, extend, reference_proportions,
                                    retarget_body, retarget_tracks)

W, H, T = 640, 480, 8
REST = {
    0: (0, .72, .08), 1: (.03, .76, .07), 2: (.045, .76, .07), 3: (.06, .76, .06), 4: (-.03, .76, .07),
    5: (-.045, .76, .07), 6: (-.06, .76, .06), 7: (.08, .74, 0), 8: (-.08, .74, 0), 9: (.03, .68, .07),
    10: (-.03, .68, .07), 11: (.18, .5, 0), 12: (-.18, .5, 0), 13: (.2, .22, 0), 14: (-.2, .22, 0),
    15: (.22, -.05, 0), 16: (-.22, -.05, 0), 17: (.23, -.12, .01), 18: (-.23, -.12, .01), 19: (.22, -.13, 0),
    20: (-.22, -.13, 0), 21: (.24, -.1, .02), 22: (-.24, -.1, .02), 23: (.1, 0, 0), 24: (-.1, 0, 0),
    25: (.1, -.42, .02), 26: (-.1, -.42, .02), 27: (.1, -.82, 0), 28: (-.1, -.82, 0), 29: (.1, -.86, -.04),
    30: (-.1, -.86, -.04), 31: (.1, -.86, .1), 32: (-.1, -.86, .1),
}
LEFT_ARM = (13, 15, 17, 19, 21)
RIGHT_LEG = (26, 28, 30, 32)


def _rot_x(a):
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def _rot_y(a):
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def _scaled_rest(scale_by_bone):
    """Rest pose with some bones scaled (child-like character), built with the same tree."""
    pts = np.array([REST[i] for i in range(33)], float)
    ext, v = extend(pts[None], np.ones((1, 33), bool))
    out = ext.copy()
    for i, (p, c) in enumerate(BONES):
        out[:, c] = out[:, p] + (ext[:, c] - ext[:, p]) * scale_by_bone.get((p, c), 1.0)
    return out[0, :33]


def _animate(rest, n=T):
    """World (T,33,3): left arm swings, right leg kicks, body yaws, root walks."""
    world = np.repeat(rest[None], n, 0).astype(float)
    for t in range(n):
        a = 0.9 * np.sin(t / 2)
        world[t, list(LEFT_ARM)] = (world[t, list(LEFT_ARM)] - world[t, 11]) @ _rot_x(a).T + world[t, 11]
        world[t, list(RIGHT_LEG)] = (world[t, list(RIGHT_LEG)] - world[t, 24]) @ _rot_x(-0.6 * a).T + world[t, 24]
        world[t] = world[t] @ _rot_y(0.3 * np.sin(t / 3)).T
    root = np.stack([np.linspace(0, 0.4, n), np.zeros(n), np.zeros(n)], -1)
    return world, root


def _project(world, root, cam_scale=220.0, cx=300.0, cy=240.0):
    """Orthographic camera; returns kp2d normalised (T,33,4) with visibility 1."""
    p = world + root[:, None]
    u = (cx + cam_scale * p[..., 0]) / W
    v = (cy - cam_scale * p[..., 1]) / H
    return np.concatenate([np.stack([u, v, p[..., 2]], -1), np.ones(world.shape[:2] + (1,))], -1)


def _track(world, root):
    kp = _project(world, root)
    return {"meta": {"format_version": 1, "fps": 16.0, "width": W, "height": H, "num_frames": len(kp), "source": "syn"},
            "kp2d": kp.astype(np.float32), "kp3d_world": world.astype(np.float32),
            "present": np.ones(len(kp), bool)}


CHILD = {(PELVIS, NECK): 0.75, (23, 25): 0.6, (25, 27): 0.6, (24, 26): 0.6, (26, 28): 0.6, (11, 13): 0.7,
         (13, 15): 0.7, (12, 14): 0.7, (14, 16): 0.7, (NECK, 0): 1.2}


def _child_reference(ref_w=400, ref_h=600):
    rest = _scaled_rest(CHILD)
    kp = np.concatenate([np.stack([(200 + 300 * rest[:, 0]) / ref_w, (380 - 300 * rest[:, 1]) / ref_h, rest[:, 2]], -1),
                         np.ones((33, 1))], -1)
    return rest, kp, reference_proportions(kp, rest, ref_w, ref_h)


def _lengths(world):
    ext, v = extend(world, np.ones(world.shape[:2], bool))
    return bone_lengths(ext, v)[0]


# --- invariants ----------------------------------------------------------------------------------------

def test_world_bone_lengths_become_the_reference_and_directions_stay_the_drivers():
    world, root = _animate(np.array([REST[i] for i in range(33)], float))
    child_rest, _, target = _child_reference()
    new, report = retarget_body(_track(world, root), target, anchor="driver")
    got = _lengths(new["kp3d_world"].astype(float))
    want = _lengths(child_rest[None])[0]
    assert np.allclose(got, want[None], atol=1e-5)  # every frame has the child's proportions
    d_ext, _ = extend(world, np.ones((T, 33), bool))
    n_ext, _ = extend(new["kp3d_world"].astype(float), np.ones((T, 33), bool))
    for p, c in BONES:  # motion = bone directions, unchanged
        a, b = d_ext[:, c] - d_ext[:, p], n_ext[:, c] - n_ext[:, p]
        cos = (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)
        assert np.allclose(cos, 1, atol=1e-6), (p, c)
    assert report["source"] == "world" and report["torso_ratio"] == pytest.approx(0.75)


def test_image_track_is_the_projection_of_the_retargeted_body():
    """Orthographic camera: retargeting in 2D == projecting the 3D retarget (foreshortening preserved)."""
    world, root = _animate(np.array([REST[i] for i in range(33)], float))
    _, _, target = _child_reference()
    track = _track(world, root)
    new, _ = retarget_body(track, target, anchor="driver")
    ext_w, _ = extend(new["kp3d_world"].astype(float), np.ones((T, 33), bool))
    ext_root, _ = extend(world, np.ones((T, 33), bool))
    # MediaPipe world is hip-centred: add the driver pelvis path back (scaled with the torso) before projecting.
    path = root + ext_root[:, PELVIS]
    expected = _project(new["kp3d_world"].astype(float), path[0] + (path - path[0]) * 0.75)
    assert np.allclose(new["kp2d"][..., :2], expected[..., :2], atol=1e-5)


def test_identity_retarget_is_a_no_op():
    world, root = _animate(np.array([REST[i] for i in range(33)], float))
    track = _track(world, root)
    own = reference_proportions(track["kp2d"][0], world[0], W, H)
    new, report = retarget_body(track, own, anchor="driver")
    assert np.allclose(new["kp2d"][..., :2], track["kp2d"][..., :2], atol=1e-6)
    assert np.allclose(report["bone_scale"], 1, atol=1e-3)


def test_anchor_places_skeleton_on_the_reference_and_scales_root_motion():
    world, root = _animate(np.array([REST[i] for i in range(33)], float))
    child_rest, ref_kp, target = _child_reference(400, 600)
    new, report = retarget_body(_track(world, root), target, anchor="reference")
    assert new["meta"]["width"] == 400 and new["meta"]["height"] == 600
    px = new["kp2d"][..., :2].astype(float) * np.array([400, 600])
    ext, _ = extend(px, np.ones((T, 33), bool))
    assert np.allclose(ext[0, PELVIS], target.pelvis_px, atol=1e-4)
    assert np.linalg.norm(ext[0, NECK] - ext[0, PELVIS]) == pytest.approx(target.torso_px, rel=1e-5)
    # root displacement measured in torso lengths is the driver's
    drv = _project(world, root)[..., :2] * np.array([W, H])
    dext, _ = extend(drv, np.ones((T, 33), bool))
    drv_disp = (dext[-1, PELVIS] - dext[0, PELVIS]) / np.linalg.norm(dext[0, NECK] - dext[0, PELVIS])
    new_disp = (ext[-1, PELVIS] - ext[0, PELVIS]) / np.linalg.norm(ext[0, NECK] - ext[0, PELVIS])
    assert np.allclose(drv_disp, new_disp, atol=1e-5)


def test_missing_joints_propagate_down_the_chain_and_are_never_invented():
    world, root = _animate(np.array([REST[i] for i in range(33)], float))
    track = _track(world, root)
    track["kp2d"][3, 13, 3] = 0.1  # left elbow not visible in frame 3
    _, _, target = _child_reference()
    new, _ = retarget_body(track, target, anchor="driver")
    vis = new["kp2d"][..., 3]
    assert (vis[3, [13, 15, 17, 19, 21]] == 0).all() and vis[3, 14] == 1 and (vis[2, [13, 15]] == 1).all()


def test_reference_missing_bones_use_the_mirror_and_torso_is_required():
    _, kp, _ = _child_reference()
    rest = _scaled_rest(CHILD)
    kp_left_leg_hidden = kp.copy()
    kp_left_leg_hidden[[25, 27, 29, 31], 3] = 0.0
    prop = reference_proportions(kp_left_leg_hidden, rest, 400, 600)
    filled = prop.filled(np.ones(len(BONES)), 1.0)
    left_shin, right_shin = BONES.index((25, 27)), BONES.index((26, 28))
    assert prop.lengths[left_shin] is None and filled[left_shin] == pytest.approx(filled[right_shin])
    assert any("not visible" in n for n in prop.notes)
    no_torso = kp.copy()
    no_torso[[23, 24], 3] = 0.0
    with pytest.raises(ValueError, match="torso"):
        reference_proportions(no_torso, rest, 400, 600)


def test_hands_and_face_follow_the_retargeted_body():
    world, root = _animate(np.array([REST[i] for i in range(33)], float))
    track = _track(world, root)
    _, _, target = _child_reference()
    hands_kp = np.zeros((T, 2, 21, 3), np.float32)
    for side, wr in ((0, 15), (1, 16)):
        hands_kp[:, side, :, :2] = track["kp2d"][:, None, wr, :2] + np.linspace(0, 0.03, 21)[None, :, None]
    face_lm = np.repeat(track["kp2d"][:, None, 0, :3], 478, 1) + np.random.default_rng(0).normal(0, 0.01, (T, 478, 3))
    tracks = {"body": track,
              "hands": {"meta": dict(track["meta"]), "kp2d": hands_kp, "present": np.ones((T, 2), bool)},
              "face": {"meta": dict(track["meta"]), "landmarks": face_lm.astype(np.float32), "present": np.ones(T, bool)}}
    new, report = retarget_tracks(tracks, target, anchor="reference")
    b, hd = new["body"]["kp2d"], new["hands"]["kp2d"]
    assert np.allclose(hd[:, 0, 0, :2], b[:, 15, :2], atol=1e-6) and np.allclose(hd[:, 1, 0, :2], b[:, 16, :2], atol=1e-6)
    drv_span = np.linalg.norm(hands_kp[:, 0, 20, :2] * [W, H] - hands_kp[:, 0, 0, :2] * [W, H], axis=-1)
    new_span = np.linalg.norm(hd[:, 0, 20, :2] * [400, 600] - hd[:, 0, 0, :2] * [400, 600], axis=-1)
    forearm = BONES.index((13, 15))
    assert np.allclose(new_span / drv_span, report["bone_scale"][forearm] * report["global_scale"], rtol=1e-3)
    assert np.allclose(new["face"]["landmarks"][:, :, :2].mean(1), b[:, 0, :2], atol=0.02)  # centred on the nose
    assert new["body"]["meta"]["retarget"]["anchor"] == "reference"
    frames = render_control_frames(new["body"], new["hands"], new["face"], 400, 600)
    assert len(frames) == T and frames[0].shape == (600, 400, 3) and frames[0].max() > 0


def test_child_is_smaller_than_adult_when_both_anchor_to_same_torso():
    """Legs are shorter relative to the torso after retargeting to child proportions."""
    world, root = _animate(np.array([REST[i] for i in range(33)], float))
    _, _, target = _child_reference()
    new, _ = retarget_body(_track(world, root), target, anchor="driver")
    ext, v = extend(new["kp3d_world"].astype(float), np.ones((T, 33), bool))
    lengths, _ = bone_lengths(ext, v)
    leg = lengths[:, BONES.index((23, 25))] + lengths[:, BONES.index((25, 27))]
    drv_ext, dv = extend(world, np.ones((T, 33), bool))
    drv_lengths, _ = bone_lengths(drv_ext, dv)
    drv_leg = drv_lengths[:, BONES.index((23, 25))] + drv_lengths[:, BONES.index((25, 27))]
    assert np.allclose(leg / lengths[:, TORSO_BONE], (drv_leg / drv_lengths[:, TORSO_BONE]) * 0.6 / 0.75, rtol=1e-5)


# --- integration: core inference and mova preprocess -------------------------------------------------

def _write_tracks(folder, track):
    import torch

    folder.mkdir(parents=True, exist_ok=True)
    t = {k: (torch.from_numpy(np.ascontiguousarray(v)) if isinstance(v, np.ndarray) else v) for k, v in track.items()}
    torch.save(t, folder / "body_motion.pt")


def test_core_inference_with_retarget(tmp_path, monkeypatch):
    import json

    import core.inference as ci
    import evaluation.identity as ident
    import preprocessing.pipeline as pipeline
    import preprocessing.retarget as rt
    from common.video_io import write_video
    from core.inference import InferenceRequest, run_inference
    from PIL import Image

    world, root = _animate(np.array([REST[i] for i in range(33)], float), n=8)
    track = _track(world, root)
    child_rest, child_kp, _ = _child_reference(64, 96)

    def fake_extract(video, out, cfg):  # stands in for MediaPipe on a full-body driver
        _write_tracks(out, track)
        return {"num_frames": T}

    monkeypatch.setattr(pipeline, "extract_motion", fake_extract)
    monkeypatch.setattr(ident, "analyze_reference", lambda rgb, *a: (None, child_kp))
    monkeypatch.setattr(rt, "_reference_world", lambda rgb: child_rest)
    monkeypatch.setattr(ci, "RUNS_DIR", tmp_path / "runs")
    ref = tmp_path / "ref.png"
    Image.fromarray(np.full((96, 64, 3), 120, np.uint8)).save(ref)
    motion = write_video(tmp_path / "motion.mp4", [np.zeros((48, 64, 3), np.uint8)] * 8, 16)
    res = run_inference(InferenceRequest(model="tiny", reference=str(ref), motion=str(motion),
                                         overrides=["inputs.retarget=true", f"output.dir={(tmp_path / 'o').as_posix()}"]),
                        echo=lambda _: None)
    assert res.stats["output_validation"]["status"] == "PASS"
    report = json.loads((res.artifacts_dir / "motion_retargeted" / "retarget.json").read_text())
    assert report["source"] == "world" and report["anchor"] == "reference"
    from common.video_io import probe_video

    assert (probe_video(res.artifacts_dir / "motion_retargeted" / "pose_openpose.mp4").width,
            probe_video(res.artifacts_dir / "motion_retargeted" / "pose_openpose.mp4").height) == (64, 96)


def test_retarget_refuses_prerendered_control(tmp_path):
    from common.errors import ConfigError
    from core.inference import build_control

    cfg = {"inputs": {"motion": str(tmp_path / "x.mp4"), "motion_is_control": True, "retarget": True}}
    with pytest.raises(ConfigError, match="raw motion video"):
        build_control(cfg, tmp_path)


def _mp_models_present():
    from preprocessing.mp_models import MODEL_DIR

    return all((MODEL_DIR / f"{n}.task").exists() for n in ("pose_landmarker_full", "face_landmarker", "hand_landmarker"))


@pytest.mark.skipif(not _mp_models_present(), reason="MediaPipe .task models not downloaded")
def test_real_reference_without_visible_hips_gives_a_clear_error(tmp_path, monkeypatch):
    skdata = pytest.importorskip("skimage.data")
    import core.preprocess as cp
    from common.errors import InvalidInputError
    from common.experiment import ExperimentRun
    from common.video_io import write_video
    from functools import partial
    from PIL import Image

    monkeypatch.setattr(cp, "ExperimentRun", partial(ExperimentRun, runs_dir=tmp_path / "runs"))
    img = skdata.astronaut()
    Image.fromarray(img).save(tmp_path / "ref.png")
    video = write_video(tmp_path / "m.mp4", [img] * 4, 8)
    with pytest.raises(InvalidInputError, match="torso"):
        cp.run_extraction(str(video), str(tmp_path / "out"), overrides=["write_previews=false"],
                          retarget_to=str(tmp_path / "ref.png"))
