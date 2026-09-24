"""Resumable generate/evaluate, global-motion metrics (motion-v2) and the visual review page."""

import copy
import json
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.benchmark as cli
from common.experiment import ExperimentRun
from common.video_io import write_video
from evaluation.motion import compare_tracks
from evaluation.protocol import freeze_manifest, verify_lock
from evaluation.video import validate_video
from tests.test_benchmark import manifest_with_media, tracks


def two_case_lock(tmp_path):
    manifest = manifest_with_media(tmp_path)
    second = copy.deepcopy(manifest["cases"][0])
    second["id"] = "test-02"
    manifest["cases"].append(second)
    lock = tmp_path / "lock.json"
    freeze_manifest(manifest, lock)
    return manifest, lock


def fake_baseline(tmp_path, manifest, calls, fail_on=None):
    def run(command, **kwargs):
        cfg_path = Path(command[command.index("--config") + 1])
        cid = cfg_path.stem
        calls.append(cid)
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text())
        run_id = f"run-{cid}-{len(calls)}"
        out = Path(cfg["output"]["dir"]) / run_id
        out.mkdir(parents=True)
        if cid == fail_on:
            raise KeyboardInterrupt  # simulated Ctrl+C / crash mid-case, before output exists
        (out / "output.mp4").write_bytes(Path(manifest["cases"][0]["motion"]).read_bytes())
        rec = tmp_path / "experiments/runs" / run_id
        rec.mkdir(parents=True)
        (rec / "run.json").write_text(json.dumps(dict(
            stats=dict(generation_time_s=2, vram_peak_gb=None), hardware={}, git_commit="test",
            model_revision="c" * 40, settings={"seed": 42, "revision": "c" * 40})))
    return run


def test_generate_resumes_after_interruption(tmp_path, monkeypatch):
    manifest, lock = two_case_lock(tmp_path)
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    locked = verify_lock(lock)
    calls = []
    with pytest.raises(KeyboardInterrupt):
        cli.generate_benchmark(locked, lock, runner=fake_baseline(tmp_path, manifest, calls, fail_on="test-02"))
    root = next((tmp_path / "outputs/benchmark").iterdir())
    state = json.loads((root / "generation.json").read_text())
    assert state["status"] == "in_progress" and list(state["cases"]) == ["test-01"]

    calls.clear()
    cli.generate_benchmark(locked, lock, resume=root, runner=fake_baseline(tmp_path, manifest, calls))
    state = json.loads((root / "generation.json").read_text())
    assert calls == ["test-02"]  # finished case was not regenerated
    assert state["status"] == "complete" and state["attempts"] == {"test-01": 1, "test-02": 2}
    assert state["cases"]["test-02"]["model_revision"] == "c" * 40
    assert not list(root.glob("*.tmp"))

    calls.clear()
    cli.generate_benchmark(locked, lock, resume=root, runner=fake_baseline(tmp_path, manifest, calls))
    assert calls == []  # already complete


def test_resume_refuses_changed_video_or_environment(tmp_path, monkeypatch):
    manifest, lock = two_case_lock(tmp_path)
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    locked = verify_lock(lock)
    with pytest.raises(KeyboardInterrupt):
        cli.generate_benchmark(locked, lock, runner=fake_baseline(tmp_path, manifest, [], fail_on="test-02"))
    root = next((tmp_path / "outputs/benchmark").iterdir())

    env = cli.generation_environment()
    monkeypatch.setattr(cli, "generation_environment",
                        lambda: {**env, "packages": {**env["packages"], "diffusers": "9.9.9"}})
    with pytest.raises(ValueError, match="diffusers"):
        cli.generate_benchmark(locked, lock, resume=root, runner=fake_baseline(tmp_path, manifest, []))
    monkeypatch.setattr(cli, "generation_environment", lambda: env)

    with open(root / "videos" / "test-01.mp4", "ab") as fh:
        fh.write(b"tampered")
    with pytest.raises(ValueError, match="changed since it was generated"):
        cli.generate_benchmark(locked, lock, resume=root, runner=fake_baseline(tmp_path, manifest, []))


def test_evaluate_resumes_and_reuses_identical_cases(tmp_path, monkeypatch):
    import evaluation.benchmark as bench

    manifest, lock = two_case_lock(tmp_path)
    monkeypatch.setattr(bench, "ExperimentRun", partial(ExperimentRun, runs_dir=tmp_path / "runs"))
    monkeypatch.setattr(bench, "evaluator_signature", lambda protocol, *_: "sig")
    monkeypatch.setattr(bench, "load_tracks", lambda folder: tracks())
    monkeypatch.setattr(bench, "frame_metrics", lambda *a, **k: {})  # identity/temporal: tested elsewhere
    extracted = []

    def fake_extract(video, out, cfg):
        extracted.append(Path(out).parent.name)
        if len(extracted) == 3:  # case 1 driver+generated done, interrupt during case 2
            raise KeyboardInterrupt

    monkeypatch.setattr(bench, "extract_motion", fake_extract)
    outputs = tmp_path / "generated"
    outputs.mkdir()
    for cid in ("test-01", "test-02"):
        (outputs / f"{cid}.mp4").write_bytes(Path(manifest["cases"][0]["motion"]).read_bytes())
    kw = dict(label="x", generation_contract={"seed": 42}, artifacts_root=tmp_path / "eval", build_review_page=False)
    with pytest.raises(KeyboardInterrupt):
        bench.evaluate_benchmark(lock, outputs, **kw)
    first = next((tmp_path / "eval").iterdir())
    partial_report = json.loads((first / "report.partial.json").read_text())
    assert [c["id"] for c in partial_report["cases"]] == ["test-01"]

    extracted.clear()
    _, report = bench.evaluate_benchmark(lock, outputs, resume_from=first, **kw)
    assert report["status"] == "complete"
    assert report["cases"][0]["reused_from"] == str(first) and "reused_from" not in report["cases"][1]
    assert extracted == ["test-02", "test-02"]

    with pytest.raises(ValueError, match="generation_contract"):
        bench.evaluate_benchmark(lock, outputs, resume_from=first, **{**kw, "generation_contract": {"seed": 1}})


def _moving_tracks(n=9):
    t = tracks(n)
    body = t["body"]["kp2d"]
    body[:, 11, :2], body[:, 12, :2] = [0.45, 0.3], [0.55, 0.3]
    body[:, 23, :2], body[:, 24, :2] = [0.46, 0.6], [0.54, 0.6]
    body[:, :, 0] += np.linspace(0, 0.2, n)[:, None]  # walks to the right
    t["body"]["meta"].update(width=64, height=64)
    t["hands"]["meta"].update(width=64, height=64)
    world = np.zeros((n, 33, 3))
    world[:, 11], world[:, 12] = [0.2, 0, 0], [-0.2, 0, 0]
    t["body"]["kp3d_world"] = world
    t["face"]["head_transform"] = np.tile(np.eye(4), (n, 1, 1))
    return t


def test_trajectory_catches_walking_in_place_that_pose_metrics_miss():
    ref = _moving_tracks()
    gen = copy.deepcopy(ref)
    gen["body"]["kp2d"][:, :, 0] -= np.linspace(0, 0.2, 9)[:, None]  # same pose, stays in place
    m = compare_tracks(ref, gen)
    assert m["body"]["pck"] == 1.0  # root-centered pose cannot see it
    assert m["trajectory"]["trajectory_error"] == pytest.approx(1 / 3)
    assert m["trajectory"]["final_displacement_error"] == pytest.approx(2 / 3)

    bigger = copy.deepcopy(ref)  # larger character following the same path, relative to its own torso
    k = bigger["body"]["kp2d"]
    k[..., :2] = 0.5 + (k[..., :2] - 0.5) * 1.2
    m = compare_tracks(ref, bigger)
    assert m["trajectory"]["trajectory_error"] == pytest.approx(0, abs=1e-9)
    assert m["trajectory"]["scale_log_error"] == pytest.approx(0, abs=1e-9)


def test_head_and_body_rotation_metrics():
    ref = _moving_tracks()
    gen = copy.deepcopy(ref)
    th = np.radians(30)
    gen["face"]["head_transform"][:, :3, :3] = [[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]]
    m = compare_tracks(ref, gen)["head_rotation"]
    assert m["geodesic_error_deg"] == pytest.approx(30)
    assert m["relative_geodesic_error_deg"] == pytest.approx(0, abs=1e-6)  # constant offset only

    turning = copy.deepcopy(ref)  # generated torso turns 45 degrees over the clip, driver does not
    ang = np.radians(np.linspace(0, 45, 9))
    turning["body"]["kp3d_world"][:, 11] = np.stack([0.2 * np.cos(ang), 0 * ang, 0.2 * np.sin(ang)], -1)
    turning["body"]["kp3d_world"][:, 12] = -turning["body"]["kp3d_world"][:, 11]
    m = compare_tracks(ref, turning)["body_orientation"]
    assert m["relative_yaw_error_deg"] == pytest.approx(22.5)

    old = tracks()  # tracks without 3D/head data stay explicit, never zero
    m = compare_tracks(old, copy.deepcopy(old))
    assert m["head_rotation"]["status"] == "UNAVAILABLE" and m["body_orientation"]["status"] == "UNAVAILABLE"


def test_review_page_with_videos_metrics_and_flags(tmp_path):
    from PIL import Image

    from evaluation.review import build_review

    w = h = 64
    Image.new("RGB", (80, 120), (200, 120, 90)).save(tmp_path / "ref.png")
    frames = [np.full((h, w, 3), 40 * i, np.uint8) for i in range(5)]
    write_video(tmp_path / "driver.mp4", frames, 16)
    write_video(tmp_path / "gen.mp4", frames, 16)
    for sub in ("driver", "generated"):
        (tmp_path / "art" / sub).mkdir(parents=True)
        t = _moving_tracks(5)["body"]
        torch.save({k: torch.as_tensor(v) if isinstance(v, np.ndarray) else v for k, v in t.items()},
                   tmp_path / "art" / sub / "body_motion.pt")
    ref, gen = _moving_tracks(5), _moving_tracks(5)
    gen["body"]["kp2d"][:, :, 0] -= np.linspace(0, 0.6, 5)[:, None]  # walks the opposite way: 1 torso off on average
    good = {"id": "walk-01", "category": "walking", "status": "evaluated", "artifacts": str(tmp_path / "art"),
            "inputs": {"reference": str(tmp_path / "ref.png"), "motion": str(tmp_path / "driver.mp4"),
                       "output": str(tmp_path / "gen.mp4")},
            "metrics": compare_tracks(ref, gen), "integrity": {"status": "PASS", "codec": "avc1"}, "generation": {}}
    bad = {"id": "fail-01", "category": "dancing", "status": "failed", "error": "ValueError: missing output"}
    report = {"label": "unit", "status": "partial", "benchmark_sha256": "b" * 64, "evaluator_sha256": "e" * 64,
              "quality_status": "NOT_ESTABLISHED", "protocol": {"width": w, "height": h, "fps": 16, "frames": 5},
              "cases": [good, bad]}
    (tmp_path / "report.json").write_text(json.dumps(report))
    index = build_review(tmp_path / "report.json")
    page = index.read_text(encoding="utf-8")
    assert "walk-01" in page and "fail-01" in page and "missing output" in page
    assert "root path off by" in page  # walking-in-place flagged
    assert validate_video(index.parent / "videos" / "walk-01.mp4", count=5, width=4 * w, height=h + 24,
                          fps=16)["status"] == "PASS"
    flags = json.loads((index.parent / "flags.json").read_text())
    assert flags["cases"]["fail-01"][0]["level"] == "error"
    rows = (index.parent / "review.csv").read_text().splitlines()
    assert len(rows) == 3 and rows[0].startswith("benchmark_sha256,case_id,auto_flags")
