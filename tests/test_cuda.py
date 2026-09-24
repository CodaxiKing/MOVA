"""GPU path, end to end, with the tiny random WanVACE model (real Diffusers code, no downloads).

Skipped on machines without a CUDA device. These are the tests that turn "pytorch/cuda" from UNVERIFIED into
verified for the wiring (device placement, offload hooks, precision); they say nothing about Wan quality.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import core.inference as core_inference
from core.inference import InferenceRequest, run_inference
from runtime import get_runtime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmark/baseline"))
from capture_tiny_vace import frames_digest, synthetic_inputs  # noqa: E402

CUDA = [d.id for d in get_runtime("pytorch").devices() if d.id.startswith("cuda")]
pytestmark = pytest.mark.skipif(not CUDA, reason="no CUDA device")


@pytest.fixture(autouse=True)
def isolated_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(core_inference, "RUNS_DIR", tmp_path / "runs")


@pytest.fixture
def media(tmp_path):
    from PIL import Image

    from common.video_io import write_video

    ref = tmp_path / "ref.png"
    Image.fromarray(np.full((48, 40, 3), (200, 120, 60), np.uint8)).save(ref)
    frames = []
    for t in range(8):
        f = np.zeros((32, 32, 3), np.uint8)
        f[8:20, 2 + 3 * t:10 + 3 * t] = 255
        frames.append(f)
    return ref, write_video(tmp_path / "control.mp4", frames, fps=16)


@pytest.mark.parametrize("offload", ["none", "model", "sequential"])
@pytest.mark.parametrize("precision", ["fp32", "bf16", "fp16"])
def test_core_end_to_end_on_cuda(media, tmp_path, precision, offload):
    ref, ctl = media
    req = InferenceRequest(model="tiny", reference=str(ref), motion=str(ctl),
                           output=str(tmp_path / f"out_{precision}_{offload}.mp4"),
                           overrides=["inputs.motion_is_control=true", f"output.dir={(tmp_path / 'out').as_posix()}",
                                      f"runtime.device={CUDA[0]}", f"runtime.precision={precision}",
                                      f"runtime.offload={offload}"])
    res = run_inference(req, echo=lambda _: None)
    assert res.context["device"] == CUDA[0] and res.context["precision"] == precision
    assert res.stats["output_validation"]["status"] == "PASS"
    rec = json.loads(res.record.read_text())
    assert rec["status"] == "success" and rec["runtime"]["offload"] == offload
    assert rec["stats"]["vram_peak_gb"] is not None  # memory is tracked on the GPU


def test_cuda_fp32_matches_cpu_numerically():
    """Same seed and inputs on CPU and GPU: not bit-exact (different kernels), but numerically the same video."""
    from common.config import load_config
    from models.registry import create_model

    rt = get_runtime("pytorch")
    outs = {}
    for dev, off in (("cpu", "none"), (CUDA[0], "model")):
        ctx = rt.context(device=dev, precision="fp32", offload=off)
        model = create_model("tiny", load_config("configs/smoke_tiny_vace.yaml"))
        model.configure(ctx, (32, 32))
        model.load(rt, ctx)
        frames, _ = model.generate(*synthetic_inputs()[:2])
        model.unload()
        assert frames_digest(frames)["finite"]
        outs[dev] = np.asarray(frames, dtype=np.float32)
    assert outs["cpu"].shape == outs[CUDA[0]].shape
    assert np.abs(outs["cpu"] - outs[CUDA[0]]).max() < 1e-3
    assert torch.cuda.memory_allocated() < 64 * 1024**2  # unload released the GPU
