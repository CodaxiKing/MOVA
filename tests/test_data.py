"""Data intake: source validation (licenses, leakage, benchmark overlap), training-manifest build, benchmark intake."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from common.video_io import write_video
from training.data_sources import load_registry, validate_sources

ROOT = Path(__file__).resolve().parent.parent


def _media(tmp_path, name, frames=20, fps=16, size=64):
    rng = np.random.default_rng(len(name))
    video = write_video(tmp_path / f"{name}.mp4", [rng.integers(0, 255, (size, size, 3), dtype=np.uint8)] * frames, fps)
    ref = tmp_path / f"{name}.png"
    Image.fromarray(rng.integers(0, 255, (48, 40, 3), dtype=np.uint8)).save(ref)
    return video.name, ref.name


def _sources(tmp_path, clips, use="research"):
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump({"schema_version": 1, "intended_use": use, "clips": clips}))
    return path


def _clip(tmp_path, cid, identity, split="train", source="own", **kw):
    video, ref = _media(tmp_path, cid)
    return {"id": cid, "identity_id": identity, "split": split, "motion": video, "references": [ref],
            "source": source, "license": "own-recording", **kw}


def test_registry_encodes_the_audit():
    reg = load_registry()
    assert reg["insightface_models"]["status"] == "blocked" and reg["liveportrait"]["status"] == "blocked"
    assert reg["unianimate_dit"]["status"] == "blocked" and reg["aistpp"]["status"] == "research_only"
    assert reg["dinov2_small"]["status"] == "allowed" and reg["dinov2_small"]["size_bytes"] == 88249960


def test_valid_sources_pass(tmp_path):
    rep = validate_sources(_sources(tmp_path, [_clip(tmp_path, "a", "p1"), _clip(tmp_path, "b", "p2", "val")]))
    assert rep["ok"], rep["problems"]


def test_every_problem_is_reported(tmp_path):
    clips = [
        _clip(tmp_path, "a", "p1"),
        _clip(tmp_path, "b", "p1", split="val"),                                      # identity leakage
        _clip(tmp_path, "c", "p3", source="aistpp", source_url="https://x"),          # research only
        _clip(tmp_path, "d", "p4", source="liveportrait", source_url="https://x"),    # blocked
        _clip(tmp_path, "e", "p5", source="humanvid"),                                # missing source_url
        _clip(tmp_path, "f", "bench-person"),                                         # benchmark identity
        _clip(tmp_path, "g", "p7", source="mystery"),                                 # not in registry
        {**_clip(tmp_path, "h", "p8"), "license": ""},                               # no per-item license
        {**_clip(tmp_path, "i", "p9"), "motion": "missing.mp4"},
    ]
    rep = validate_sources(_sources(tmp_path, clips, use="commercial"), forbid_identities={"bench-person"})
    text = "\n".join(rep["problems"])
    for needle in ("identity p1 appears in ['train', 'val']", "c: source 'aistpp' is 'research_only'",
                   "d: source 'liveportrait' is 'blocked'", "e: source_url required", "f: identity bench-person",
                   "g: source 'mystery' is not in", "h: missing license", "i: file not found missing.mp4"):
        assert needle in text, needle
    assert not rep["ok"]


def test_short_video_is_rejected(tmp_path):
    c = _clip(tmp_path, "a", "p1")
    write_video(tmp_path / c["motion"], [np.zeros((32, 32, 3), np.uint8)] * 5, 16)
    assert "a: 5 frames < 17" in validate_sources(_sources(tmp_path, [c]))["problems"]


def _mp_models_present():
    from preprocessing.mp_models import MODEL_DIR

    return all((MODEL_DIR / f"{n}.task").exists() for n in ("pose_landmarker_full", "face_landmarker", "hand_landmarker"))


@pytest.mark.skipif(not _mp_models_present(), reason="MediaPipe .task models not downloaded")
def test_build_manifest_feeds_the_training_dataset(tmp_path):
    pytest.importorskip("diffusers")
    import torch
    from diffusers import AutoencoderKLWan

    from training.data_sources import build_training_manifest
    from training.dataset import MotionClipDataset

    src = _sources(tmp_path, [_clip(tmp_path, "a", "p1"), _clip(tmp_path, "b", "p2")])
    torch.manual_seed(0)
    vae = AutoencoderKLWan(base_dim=3, z_dim=16, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
                           temperal_downsample=[False, True, True]).eval()
    manifest = build_training_manifest(src, tmp_path / "built", num_frames=9, size=32, vae=vae,
                                       encode_prompt=lambda p: torch.zeros(8, 32), stride=8)
    data = json.loads(manifest.read_text())
    assert [s["id"] for s in data["samples"]] == ["a@0", "a@8", "b@0", "b@8"]   # 20 frames, windows of 9, stride 8
    assert all(s["license"] == "own-recording" for s in data["samples"])
    ds = MotionClipDataset(manifest, reference_size=32, max_references=1)
    item = ds[0]
    assert item["latents"].shape == (16, 3, 4, 4) and item["body"].shape == (9, 33, 6)
    (tmp_path / "x").mkdir()
    bad = _sources(tmp_path / "x", [_clip(tmp_path / "x", "c", "p3", source="liveportrait", source_url="u")])
    with pytest.raises(ValueError, match="invalid"):
        build_training_manifest(bad, tmp_path / "b2", num_frames=9, size=32, vae=vae, encode_prompt=lambda p: torch.zeros(8, 32))


def _bench_manifest(tmp_path):
    src = yaml.safe_load((ROOT / "benchmark/v1.draft.yaml").read_text(encoding="utf-8"))
    src["protocol"].update(width=32, height=32, frames=5, fps=16)
    case = dict(src["cases"][0])
    case.update(reference=str(tmp_path / "bench/walking-01/reference.png"),
                motion=str(tmp_path / "bench/walking-01/motion.mp4"))
    src["cases"] = [case]
    path = tmp_path / "manifest.yaml"
    path.write_text("# test manifest header\n# second line\n" + yaml.safe_dump(src, sort_keys=False), encoding="utf-8")
    return path


def test_benchmark_intake_places_media_fills_metadata_and_allows_freeze(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import benchmark as bench_cli

    from evaluation.protocol import freeze_manifest, read_manifest

    manifest = _bench_manifest(tmp_path)
    raw = write_video(tmp_path / "raw.mp4", [np.full((40, 60, 3), 90, np.uint8)] * 12, 16)
    Image.fromarray(np.full((50, 30, 3), 200, np.uint8)).save(tmp_path / "ref_in.jpg")
    out = bench_cli.intake_case(manifest, "walking-01", reference=str(tmp_path / "ref_in.jpg"), motion=str(raw),
                                identity_id="volunteer-01", reference_source="own photo",
                                motion_source="own recording", reference_license="CC-BY-4.0",
                                motion_license="CC-BY-4.0")
    assert out["remaining_issues"] == [] and out["motion_validation"]["status"] == "PASS"
    assert manifest.read_text(encoding="utf-8").startswith("# test manifest header\n# second line\n")
    m = read_manifest(manifest)
    assert m["cases"][0]["identity_id"] == "volunteer-01"
    locked = freeze_manifest(m, tmp_path / "lock.json")
    assert len(locked["benchmark_sha256"]) == 64
    with pytest.raises(FileExistsError):
        bench_cli.intake_case(manifest, "walking-01", reference=str(tmp_path / "ref_in.jpg"), motion=str(raw))
    with pytest.raises(ValueError, match="No case"):
        bench_cli.intake_case(manifest, "nope", reference=str(tmp_path / "ref_in.jpg"), motion=str(raw))


def test_intake_rejects_a_clip_that_is_too_short_and_leaves_nothing_behind(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import benchmark as bench_cli

    manifest = _bench_manifest(tmp_path)
    short = write_video(tmp_path / "short.mp4", [np.zeros((32, 32, 3), np.uint8)] * 3, 16)
    Image.fromarray(np.zeros((32, 32, 3), np.uint8)).save(tmp_path / "r.png")
    with pytest.raises(ValueError, match="too short"):
        bench_cli.intake_case(manifest, "walking-01", reference=str(tmp_path / "r.png"), motion=str(short))
    assert not (tmp_path / "bench/walking-01/reference.png").exists()
    assert "identity_id: null" in manifest.read_text(encoding="utf-8")
