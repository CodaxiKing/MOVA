"""identity-v1, temporal-v1 and blind GSB scoring, on synthetic fixtures (CPU, no downloads)."""

import csv
import json

import cv2
import numpy as np
import pytest

from evaluation.gsb import AXES, build_gsb_session, gsb_score, score_gsb
from evaluation.identity import (HFImageEmbedder, RIGID_FACE_POINTS, color_distance, color_stats,
                                 face_geometry_signature, geometry_error, identity_metrics, load_embedder)
from evaluation.review import case_flags
from evaluation.temporal import temporal_metrics, video_temporal_stats


def _face_landmarks(rng, stretch=1.0):
    """Random 3D face-like cloud in normalised image coordinates around (0.5, 0.5)."""
    pts = rng.normal(0, 0.05, (478, 3))
    pts[:, 0] *= stretch
    return pts + np.array([0.5, 0.45, 0.0])


def _rotate(lm, deg, scale=1.0, shift=(0.0, 0.0)):
    a = np.radians(deg)
    r = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])  # yaw
    c = lm.mean(0)
    return (lm - c) @ r.T * scale + c + np.array([*shift, 0.0])


# --- identity: geometry --------------------------------------------------------------------------------

def test_geometry_is_invariant_to_pose_scale_and_framing():
    rng = np.random.default_rng(0)
    lm = _face_landmarks(rng)
    ref = face_geometry_signature(lm, 512, 512)
    moved = face_geometry_signature(_rotate(lm, 25, scale=0.6, shift=(0.1, -0.05)), 512, 512)
    assert geometry_error(ref, moved) < 1e-9


def test_geometry_detects_morphology_drift():
    rng = np.random.default_rng(1)
    lm = _face_landmarks(rng)
    ref = face_geometry_signature(lm, 512, 512)
    wide = lm.copy()
    wide[:, 0] = (wide[:, 0] - 0.5) * 1.3 + 0.5
    assert geometry_error(ref, face_geometry_signature(wide, 512, 512)) > 0.05


def test_geometry_ignores_expression_points():
    rng = np.random.default_rng(2)
    lm = _face_landmarks(rng)
    smile = lm.copy()
    moving = [i for i in (13, 14, 61, 291, 0, 17, 152, 70, 300) if i not in RIGID_FACE_POINTS]
    smile[moving] += rng.normal(0, 0.03, (len(moving), 3))  # lips, mouth corners, chin, brows move
    assert geometry_error(face_geometry_signature(lm, 256, 256), face_geometry_signature(smile, 256, 256)) < 1e-9


def test_geometry_rejects_invalid_landmarks():
    lm = np.full((478, 3), np.nan)
    assert face_geometry_signature(lm, 64, 64) is None
    assert face_geometry_signature(np.zeros((10, 3)), 64, 64) is None


# --- identity: colour and full metric ---------------------------------------------------------------

def _textured(rng, hue_shift=0):
    noise = cv2.GaussianBlur((rng.random((96, 96)) * 255).astype(np.uint8), (0, 0), 3).astype(float)
    tone = (noise - noise.min()) / max(np.ptp(noise), 1)
    base = np.array([200, 90, 60], float)  # saturated skin/orange tone with texture
    img = np.clip(base * (0.6 + 0.6 * tone[..., None]), 0, 255).astype(np.uint8)
    if hue_shift:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv[..., 0] = (hsv[..., 0].astype(int) + hue_shift) % 180
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img


def test_color_distance_same_vs_recoloured():
    rng = np.random.default_rng(3)
    img = _textured(rng)
    mask = np.ones(img.shape[:2], bool)
    ref = color_stats(img, mask)
    de_same, hist_same = color_distance(ref, color_stats(img, mask))
    de_shift, hist_shift = color_distance(ref, color_stats(_textured(np.random.default_rng(3), 60), mask))
    assert de_same == pytest.approx(0) and hist_same == pytest.approx(1)
    assert de_shift > 10 and hist_shift < hist_same


def _tracks(lm, n, present):
    face = {"landmarks": np.repeat(lm[None], n, 0), "present": np.asarray(present, bool)}
    body_kp = np.zeros((33, 4))
    for i, (x, y) in zip((11, 12, 24, 23), ((0.3, 0.3), (0.7, 0.3), (0.7, 0.8), (0.3, 0.8))):
        body_kp[i] = (x, y, 0, 1)
    body = {"kp2d": np.repeat(body_kp[None], n, 0), "present": np.ones(n, bool)}
    return face, body, body_kp


def test_identity_metrics_coverage_nulls_and_units():
    rng = np.random.default_rng(4)
    lm = _face_landmarks(rng)
    img = _textured(rng)
    face, body, body_kp = _tracks(lm, 4, [1, 1, 0, 1])
    m = identity_metrics(img, lm, body_kp, [img] * 4, face, body)
    assert m["face_coverage"] == 0.75 and m["face_geometry_error"]["frames"] == 3
    assert m["face_geometry_error"]["mean"] == pytest.approx(0, abs=1e-9)
    assert m["face_color_delta_e"]["mean"] == pytest.approx(0, abs=1e-9)
    assert m["torso_coverage"] == 1.0 and m["torso_color_hist_intersection"]["mean"] == pytest.approx(1)
    assert m["embedding"]["status"] == "UNAVAILABLE"
    json.dumps(m, allow_nan=False)
    # no face in the reference: face metrics are null, never zero
    m = identity_metrics(img, None, None, [img] * 4, face, body)
    assert m["face_coverage"] is None and m["face_geometry_error"]["mean"] is None
    assert m["torso_coverage"] is None


def test_identity_flags_drift_but_not_same_identity():
    rng = np.random.default_rng(5)
    lm = _face_landmarks(rng)
    img = _textured(rng)
    face, body, body_kp = _tracks(lm, 3, [1, 1, 1])
    same = identity_metrics(img, lm, body_kp, [img] * 3, face, body)
    recol = identity_metrics(img, lm, body_kp, [_textured(np.random.default_rng(5), 60)] * 3, face, body)
    base = {"status": "evaluated", "metrics": {}}
    assert not [f for f in case_flags({**base, "metrics": {"identity": same}}) if f["level"] == "warn"]
    msgs = [f["message"] for f in case_flags({**base, "metrics": {"identity": recol}})]
    assert any("face colour drift" in x for x in msgs) and any("outfit colour drift" in x for x in msgs)


def test_embedding_backend_with_tiny_random_dinov2():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.Dinov2Config(hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
                                    image_size=32, patch_size=8)
    import torch

    torch.manual_seed(0)
    emb = HFImageEmbedder(transformers.Dinov2Model(cfg), image_size=32, name="tiny-dinov2")
    rng = np.random.default_rng(6)
    lm = _face_landmarks(rng)
    img = _textured(rng)
    face, body, body_kp = _tracks(lm, 2, [1, 1])
    m = identity_metrics(img, lm, body_kp, [img, img], face, body, embedder=emb)
    assert m["embedding"]["backend"] == "tiny-dinov2" and m["embedding"]["mean"] == pytest.approx(1, abs=1e-5)


def test_embedder_is_never_downloaded():
    embedder, reason = load_embedder("hf:mova-test/definitely-not-cached-model")
    assert embedder is None and "not available locally" in reason
    assert load_embedder(None) == (None, None)


# --- temporal ----------------------------------------------------------------------------------------

def _scene(n=7):
    rng = np.random.default_rng(7)
    tex = cv2.GaussianBlur((rng.random((96, 140, 3)) * 255).astype(np.uint8), (0, 0), 2)
    tex = cv2.normalize(tex, None, 0, 255, cv2.NORM_MINMAX)
    return [np.ascontiguousarray(np.roll(tex, 2 * t, axis=1)[:, 10:106]) for t in range(n)], rng


def test_temporal_separates_smooth_noisy_flicker_frozen():
    smooth, rng = _scene()
    noisy = [np.clip(f + rng.normal(0, 12, f.shape), 0, 255).astype(np.uint8) for f in smooth]
    flick = [np.clip(f.astype(int) + (20 if t % 2 else -20), 0, 255).astype(np.uint8) for t, f in enumerate(smooth)]
    s, n, f = (video_temporal_stats(v) for v in (smooth, noisy, flick))
    assert s["mean_flow_px"] == pytest.approx(2.0, abs=0.3)  # recovers the 2 px/frame motion
    assert n["warp_error"] > 10 * s["warp_error"] and f["warp_error"] > 10 * s["warp_error"]
    assert f["luma_flicker"] > 100 * s["luma_flicker"]
    frozen = video_temporal_stats([smooth[0]] * 5)
    assert frozen["warp_error"] < 1e-3 and frozen["mean_flow_px"] < 0.05  # remap interpolation noise only


def test_temporal_ratios_against_driver_and_json_safe():
    smooth, rng = _scene()
    noisy = [np.clip(f + rng.normal(0, 12, f.shape), 0, 255).astype(np.uint8) for f in smooth]
    m = temporal_metrics(noisy, driver=smooth)
    assert m["warp_error_ratio"] > 5 and m["mean_flow_px_ratio"] == pytest.approx(1, abs=0.2)
    frozen = temporal_metrics([smooth[0]] * 7, driver=smooth)
    assert frozen["mean_flow_px_ratio"] < 0.05
    msgs = [x["message"] for x in case_flags({"status": "evaluated", "metrics": {"temporal": frozen}})]
    assert any("frozen" in x for x in msgs)
    json.dumps(m, allow_nan=False)
    assert temporal_metrics([smooth[0]])["generated"]["warp_error"] is None


# --- GSB ------------------------------------------------------------------------------------------------

def test_gsb_formula():
    assert gsb_score(5, 3, 2) == pytest.approx(8 / 5)
    assert gsb_score(4, 0, 0) is None


def _report(tmp_path, label, value):
    from common.video_io import write_video

    cases = []
    for cid in ("a", "b", "c"):
        video = write_video(tmp_path / label / f"{cid}.mp4", [np.full((32, 32, 3), value, np.uint8)] * 5, 16)
        cases.append({"id": cid, "status": "evaluated", "inputs": {"output": str(video), "reference": None,
                                                                   "motion": None}})
    path = tmp_path / f"{label}.json"
    path.write_text(json.dumps({"label": label, "benchmark_sha256": "x", "protocol": {"width": 32, "height": 32,
                                                                                       "fps": 16}, "cases": cases}))
    return path


def test_gsb_session_is_blind_and_scores_from_candidate_view(tmp_path):
    base, cand = _report(tmp_path, "base", 40), _report(tmp_path, "cand", 200)
    paths = build_gsb_session(base, cand, tmp_path / "gsb", seed=3)
    assert "candidate" not in paths["index"].read_text() and "baseline" not in paths["ratings"].read_text()
    key = json.loads(paths["key"].read_text())
    rows = []
    for cid, side in key["cases"].items():  # a rater who always prefers the candidate on every axis
        pick = "L" if side["left"] == "candidate" else "R"
        rows.append([cid, "r1", *([pick] * len(AXES)), ""])
    rows.append(["a", "r2", *(["S"] * len(AXES)), ""])
    rows.append(["b", "r2", *([""] * len(AXES)), ""])
    with paths["ratings"].open("w", newline="") as fh:
        csv.writer(fh).writerows([["case_id", "rater", *AXES, "notes"], *rows])
    res = score_gsb(paths["ratings"], paths["key"])
    ax = res["axes"]["identity_preservation"]
    assert (ax["G"], ax["S"], ax["B"], ax["unrated"]) == (3, 1, 0, 1)
    assert ax["gsb"] == pytest.approx(4.0) and ax["win_rate"] == 1.0 and res["raters"] == ["r1", "r2"]
    with pytest.raises(FileExistsError):
        build_gsb_session(base, cand, tmp_path / "gsb")


def test_gsb_rejects_bad_ratings_and_incomparable_reports(tmp_path):
    base, cand = _report(tmp_path, "base", 40), _report(tmp_path, "cand", 200)
    paths = build_gsb_session(base, cand, tmp_path / "gsb")
    with paths["ratings"].open("w", newline="") as fh:
        csv.writer(fh).writerows([["case_id", "rater", *AXES, "notes"], ["a", "r", "X", *([""] * (len(AXES) - 1)), ""]])
    with pytest.raises(ValueError, match="not L, S, R"):
        score_gsb(paths["ratings"], paths["key"])
    other = json.loads(cand.read_text())
    other["benchmark_sha256"] = "y"
    cand.write_text(json.dumps(other))
    with pytest.raises(ValueError, match="not comparable"):
        build_gsb_session(base, cand, tmp_path / "gsb2")


# --- real MediaPipe on a public-domain photo ------------------------------------------------------

def _mp_models_present():
    from preprocessing.mp_models import MODEL_DIR

    return all((MODEL_DIR / f"{n}.task").exists() for n in ("pose_landmarker_full", "face_landmarker", "hand_landmarker"))


@pytest.mark.skipif(not _mp_models_present(), reason="MediaPipe .task models not downloaded")
def test_identity_on_real_face_detects_stretch_and_recolour(tmp_path):
    skdata = pytest.importorskip("skimage.data")
    import torch

    from common.video_io import read_video, write_video
    from evaluation.identity import analyze_reference
    from preprocessing.pipeline import ExtractionConfig, extract_motion

    img = skdata.astronaut()
    ref_face, ref_body = analyze_reference(img)
    assert ref_face is not None

    def rot(im, a):
        return cv2.warpAffine(im, cv2.getRotationMatrix2D((256, 256), a, 1.0), (512, 512), borderMode=cv2.BORDER_REFLECT)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[..., 0] = (hsv[..., 0].astype(int) + 60) % 180
    variants = {"same": img, "wide": cv2.resize(img, (660, 512))[:, 74:586],
                "recolor": cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)}
    res = {}
    for name, base in variants.items():
        video = write_video(tmp_path / f"{name}.mp4", [rot(base, 5 * np.sin(i / 2)) for i in range(5)], fps=8)
        extract_motion(video, tmp_path / name, ExtractionConfig(write_previews=False, write_openpose=False))
        ft = torch.load(tmp_path / name / "face_motion.pt", weights_only=True)
        bt = torch.load(tmp_path / name / "body_motion.pt", weights_only=True)
        res[name] = identity_metrics(img, ref_face, ref_body, read_video(video),
                                     {k: np.asarray(ft[k]) for k in ("landmarks", "present")},
                                     {k: np.asarray(bt[k]) for k in ("kp2d", "present")})
    geo = {k: v["face_geometry_error"]["mean"] for k, v in res.items()}
    de = {k: v["face_color_delta_e"]["mean"] for k, v in res.items()}
    assert res["same"]["face_coverage"] >= 0.8
    assert geo["wide"] > 3 * geo["same"] and geo["same"] < 0.035 < geo["wide"]
    assert de["recolor"] > 5 * de["same"]
