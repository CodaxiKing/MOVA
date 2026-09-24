"""Signal cleaning: One-Euro filter, hand post-processing, control rendering from tracks, EXP-003 and calibration."""

import numpy as np
import pytest

from preprocessing.filters import jitter, one_euro
from preprocessing.hands_post import fill_short_gaps, fix_side_swaps, postprocess_hands, reject_far_hands


# --- One-Euro --------------------------------------------------------------------------------------

def test_one_euro_removes_jitter_on_a_still_point():
    rng = np.random.default_rng(0)
    x = 0.5 + rng.normal(0, 0.004, (64, 1, 2))
    y = one_euro(x, 16, min_cutoff=1.0, beta=10.0)
    assert jitter(y) < 0.4 * jitter(x)


def test_higher_beta_means_less_lag_on_fast_motion():
    t = np.arange(48) / 16
    x = np.stack([0.5 + 0.15 * np.sin(4 * np.pi * t), np.full_like(t, 0.5)], -1)[:, None]
    err = {b: np.abs(one_euro(x, 16, min_cutoff=1.0, beta=b) - x).mean() for b in (0.0, 10.0, 100.0)}
    assert err[100.0] < err[10.0] < err[0.0]


def test_one_euro_resets_after_gaps_and_keeps_missing_frames_untouched():
    x = np.zeros((6, 2, 2))
    x[3:] = 1.0                                  # jump right after a gap
    present = np.array([[1, 1], [1, 1], [0, 1], [1, 1], [1, 1], [1, 1]], bool)
    y = one_euro(x, 16, min_cutoff=0.5, beta=0.0, present=present)
    assert np.allclose(y[3, 0], 1.0)             # point 0: fresh segment after the gap starts at the observation
    assert y[3, 1, 0] < 0.9                      # point 1 was continuous: filtered, lags behind the jump
    assert np.allclose(y[2, 0], x[2, 0])         # the missing sample is passed through, never invented
    with pytest.raises(ValueError):
        one_euro(x, 0)


# --- hands ---------------------------------------------------------------------------------------

def _body(t, visible=1.0):
    kp = np.zeros((t, 33, 4), np.float32)
    kp[:, 13, :2], kp[:, 15, :2] = (0.30, 0.40), (0.30, 0.50)   # left elbow/wrist
    kp[:, 14, :2], kp[:, 16, :2] = (0.70, 0.40), (0.70, 0.50)
    kp[..., 3] = visible
    return kp


def _hands(t):
    kp = np.zeros((t, 2, 21, 3), np.float32)
    kp[:, 0, :, :2] = (0.30, 0.52)
    kp[:, 1, :, :2] = (0.70, 0.52)
    return kp


def test_reject_hand_far_from_its_wrist():
    kp, pres = _hands(4), np.ones((4, 2), bool)
    kp[2, 0, :, :2] = (0.9, 0.9)                 # a "left hand" in the corner
    out, n = reject_far_hands(kp, pres, _body(4), 100, 100)
    assert n == 1 and not out[2, 0] and out[:, 1].all()


def test_side_swaps_are_fixed_by_continuity_when_body_wrists_are_unreliable():
    t = 8
    kp, pres = _hands(t), np.ones((t, 2), bool)
    kp[3:5] = kp[3:5, ::-1]                      # handedness label flipped for two frames
    source, swaps = fix_side_swaps(kp, pres, _body(t, visible=0.1))
    assert swaps == 2 and (source[3:5] == [1, 0]).all() and (source[:3] == [0, 1]).all()
    trusted, n = fix_side_swaps(kp, pres, _body(t, visible=1.0))   # confident body wrists: left as is
    assert n == 0


def test_single_hand_moves_to_the_side_it_belongs_to():
    t = 6
    kp, pres = _hands(t), np.ones((t, 2), bool)
    pres[4] = [True, False]
    kp[4, 0] = kp[3, 1]                          # the right hand was labelled "left"
    source, swaps = fix_side_swaps(kp, pres, None)
    assert swaps == 1 and source[4].tolist() == [-1, 0]


def test_fill_short_gaps_only_inside_and_only_short():
    x = np.arange(10, dtype=float)[:, None]
    present = np.array([1, 0, 0, 1, 0, 0, 0, 0, 1, 0], bool)
    y, filled = fill_short_gaps(x, present, max_gap=2)
    assert filled.tolist() == [False, True, True, False, False, False, False, False, False, False]
    assert y[1, 0] == pytest.approx(1.0) and y[5, 0] == x[5, 0]


def test_postprocess_keeps_raw_and_reports():
    t = 8
    kp, pres = _hands(t), np.ones((t, 2), bool)
    kp[3:5] = kp[3:5, ::-1]
    pres[6, 1] = False
    k3 = np.zeros((t, 2, 21, 3), np.float32)
    k3[:, 1] = 1.0
    k3[3:5] = k3[3:5, ::-1]
    hands = {"kp2d": kp, "kp3d_world": k3, "present": pres, "handedness_score": np.ones((t, 2), np.float32)}
    out, rep = postprocess_hands(hands, _body(t, visible=0.1), 100, 100)
    assert rep["side_swaps_fixed"] == 2 and rep["frames_filled_for_control"] == 1
    assert np.array_equal(out["kp2d_raw"], kp) and np.array_equal(out["present_raw"], pres)
    right = out["present"][:, 1]
    assert (out["kp3d_world"][right, 1] == 1).all() and (out["kp3d_world"][out["present"][:, 0], 0] == 0).all()
    assert not out["present"][6, 1] and out["filled"][6, 1]   # gap filled for the control, not "detected"


# --- pipeline integration -------------------------------------------------------------------------

def _mp_models_present():
    from preprocessing.mp_models import MODEL_DIR

    return all((MODEL_DIR / f"{n}.task").exists() for n in ("pose_landmarker_full", "face_landmarker", "hand_landmarker"))


@pytest.mark.skipif(not _mp_models_present(), reason="MediaPipe .task models not downloaded")
def test_extraction_writes_smoothed_features_and_control(tmp_path):
    skdata = pytest.importorskip("skimage.data")
    import cv2
    import torch

    from common.video_io import read_video, write_video
    from preprocessing.control import render_control_frames
    from preprocessing.pipeline import ExtractionConfig, extract_motion

    img = skdata.astronaut()
    frames = [cv2.warpAffine(img, cv2.getRotationMatrix2D((256, 256), 6 * np.sin(i / 2), 1.0), (512, 512),
                             borderMode=cv2.BORDER_REFLECT) for i in range(8)]
    video = write_video(tmp_path / "v.mp4", frames, 16)
    s = extract_motion(video, tmp_path / "raw", ExtractionConfig(write_previews=False, hand_postprocess=False))
    body = torch.load(tmp_path / "raw" / "body_motion.pt", weights_only=True)
    face = torch.load(tmp_path / "raw" / "face_motion.pt", weights_only=True)
    hands = torch.load(tmp_path / "raw" / "hand_motion.pt", weights_only=True)
    expected = write_video(tmp_path / "expected.mp4", render_control_frames(body, hands, face, 512, 512), 16)
    assert np.array_equal(np.asarray(read_video(tmp_path / "raw" / "pose_openpose.mp4")), np.asarray(read_video(expected)))
    assert s["control"] == {"smoothing": "none", "hand_postprocess": False}

    s = extract_motion(video, tmp_path / "smooth", ExtractionConfig(write_previews=False, smoothing="one_euro"))
    body = torch.load(tmp_path / "smooth" / "body_motion.pt", weights_only=True)
    assert body["features"]["kp2d_smoothed"].shape == (8, 33, 2) and "hands_postprocess" in s
    assert torch.equal(body["kp2d"], torch.load(tmp_path / "raw" / "body_motion.pt", weights_only=True)["kp2d"])  # raw kept
    with pytest.raises(ValueError, match="smoothing"):
        extract_motion(video, tmp_path / "bad", ExtractionConfig(write_previews=False, smoothing="kalman"))


@pytest.mark.skipif(not _mp_models_present(), reason="MediaPipe .task models not downloaded")
def test_min_confidence_sweep_reports_every_threshold(tmp_path):
    skdata = pytest.importorskip("skimage.data")
    from common.video_io import write_video
    from preprocessing.calibration import sweep_min_confidence

    video = write_video(tmp_path / "v.mp4", [skdata.astronaut()] * 5, 8)
    rep = sweep_min_confidence(video, (0.4, 0.6))
    assert [r["min_confidence"] for r in rep["rows"]] == [0.4, 0.6]
    assert rep["recommended_min_confidence"] in (0.4, 0.6) and rep["rows"][0]["body_detection"] > 0.5


# --- EXP-003 -------------------------------------------------------------------------------------

def test_rot6d_of_rest_pose_is_identity_and_rows_are_rotations():
    from evaluation.representations import REST, bone_rot6d

    rest6 = bone_rot6d(REST[None], REST)
    assert np.allclose(rest6.reshape(-1, 6), [1, 0, 0, 1, 0, 0], atol=1e-9)
    moved = REST.copy()
    moved[13:16] += 0.05
    cols = bone_rot6d(moved[None], REST).reshape(-1, 3, 2)
    assert np.allclose(np.einsum("nij,nik->njk", cols, cols), np.eye(2), atol=1e-9)  # orthonormal columns


def test_exp003_ranks_as_documented():
    from evaluation.representations import run

    r = run(seed=0, frames=32)["results"]
    assert r["2d_normalized"]["nuisance_camera"] < r["2d_image"]["nuisance_camera"]
    assert r["rot6d_limbs"]["nuisance_camera"] == pytest.approx(0, abs=1e-6)
    assert r["rot6d_limbs"]["nuisance_body_size"] == pytest.approx(0, abs=1e-6)
    assert r["rot6d_limbs"]["noise_jitter_ratio"] < 0.25 * r["rot6d_bones"]["noise_jitter_ratio"]
    assert r["3d_world_normalized"]["nuisance_body_size"] < r["3d_world"]["nuisance_body_size"]
