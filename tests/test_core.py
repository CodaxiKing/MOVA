import json

import numpy as np
import pytest
from PIL import Image

from common.config import apply_override
from common.env import HardwareInfo, select_profile
from common.experiment import STANDARD_FIELDS, ExperimentRun
from common.video_io import probe_video, read_video, write_video
from inference.conditioning import fit_resolution, letterbox, prepare_control_frames, valid_num_frames


def _hw(cuda: bool, vram: float | None = None, bf16: bool = True) -> HardwareInfo:
    return HardwareInfo(python="3.12", platform="test", torch_version="x", torch_cuda_build=None,
                        cuda_available=cuda, device="cuda" if cuda else "cpu", vram_total_gb=vram, bf16_supported=bf16)


@pytest.mark.parametrize("cuda,vram,name,offload,frames", [
    (False, None, "cpu", "none", 9),
    (True, 6.0, "cuda-lt6gb", "sequential", 9),
    (True, 8.0, "cuda-8gb", "model", 17),
    (True, 12.0, "cuda-12-16gb", "model", 33),
    (True, 24.0, "cuda-24gb+", "none", 81),
])
def test_select_profile(cuda, vram, name, offload, frames):
    p = select_profile(_hw(cuda, vram))
    assert (p.name, p.offload, p.num_frames) == (name, offload, frames)
    assert (p.num_frames - 1) % 4 == 0


def test_profile_fp16_without_bf16():
    assert select_profile(_hw(True, 8.0, bf16=False)).dtype == "float16"


def test_apply_override_nested_and_typed():
    cfg = {"a": {"b": 1}}
    apply_override(cfg, "a.b=2.5")
    apply_override(cfg, "a.c.d=true")
    apply_override(cfg, "x=null")
    assert cfg == {"a": {"b": 2.5, "c": {"d": True}}, "x": None}
    with pytest.raises(ValueError):
        apply_override(cfg, "novalue")


def test_experiment_run_writes_standard_fields(tmp_path):
    run = ExperimentRun("unit", runs_dir=tmp_path / "runs", artifacts_root=tmp_path / "out")
    run.log(model="m", frames=17)
    run.finish("success", loss=0.1)
    rec = json.loads((run.dir / "run.json").read_text())
    assert all(k in rec for k in STANDARD_FIELDS)
    assert rec["status"] == "success" and rec["frames"] == 17 and rec["loss"] == 0.1
    assert run.artifacts_dir.exists() and run.artifacts_dir != run.dir


@pytest.mark.parametrize("n,expected", [(5, 5), (8, 5), (17, 17), (20, 17), (81, 81)])
def test_valid_num_frames(n, expected):
    assert valid_num_frames(n) == expected


def test_valid_num_frames_too_short():
    with pytest.raises(ValueError):
        valid_num_frames(4)


def test_fit_resolution_multiple_and_area():
    w, h = fit_resolution(720, 1280, 256 * 256)
    assert w % 16 == 0 and h % 16 == 0 and w * h <= 256 * 256 and h > w


def test_letterbox_and_control_frames():
    img = Image.new("RGB", (100, 300), (10, 20, 30))
    out = letterbox(img, 256, 256)
    assert out.size == (256, 256) and out.getpixel((0, 128)) == (255, 255, 255)
    frames = [np.full((64, 48, 3), i, np.uint8) for i in range(10)]
    ctrl = prepare_control_frames(frames, 32, 32, 17)
    assert len(ctrl) == 17 and ctrl[0].size == (32, 32)


def test_video_roundtrip(tmp_path):
    frames = [np.full((64, 80, 3), i * 20, np.uint8) for i in range(8)]
    p = write_video(tmp_path / "v.mp4", frames, fps=8)
    meta = probe_video(p)
    back = read_video(p)
    assert (meta.width, meta.height) == (80, 64) and len(back) == 8
    assert abs(int(back[5].mean()) - 100) < 6
