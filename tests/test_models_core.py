"""Model interface + registry, capability validation, run-config precedence, core inference service and CLI.

End-to-end tests use the tiny random WanVACE model: the real Diffusers code path, no downloads, output is noise.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import core.inference as core_inference
from common.errors import (ConfigError, DeviceNotSupportedError, InsufficientResourcesError, InsufficientVRAMError,
                           InvalidInputError, ModelNotFoundError, ModelRuntimeIncompatibleError,
                           ModelWeightsMissingError, PrecisionNotSupportedError, RuntimeNotAvailableError)
from common.resources import ResourceCheck, enforce
from common.video_io import write_video
from core.capabilities import compatibility_matrix, validate
from core.config import resolve_run_config
from core.inference import InferenceRequest, plan_inference, run_inference
from models.base import ModelSpec, MotionModel, WeightsStatus
from models.registry import create_model, get_spec, model_names, register_model, resolve_name, unregister_model
from mova.cli import main as cli
from runtime import get_runtime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmark/baseline"))
from capture_tiny_vace import frames_digest, synthetic_inputs  # noqa: E402

BASELINE_SHA = json.loads((ROOT / "benchmark/baseline/tiny_vace_cpu.json").read_text())["runs"][0]["output"]["sha256"]


@pytest.fixture
def media(tmp_path):
    """Reference image and an already-rendered control video (skips MediaPipe)."""
    ref = tmp_path / "ref.png"
    Image.fromarray(np.full((48, 40, 3), (200, 120, 60), np.uint8)).save(ref)
    frames = []
    for t in range(8):
        f = np.zeros((32, 32, 3), np.uint8)
        f[8:20, 2 + 3 * t:10 + 3 * t] = 255
        frames.append(f)
    ctl = write_video(tmp_path / "control.mp4", frames, fps=16)
    return ref, ctl


@pytest.fixture(autouse=True)
def isolated_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(core_inference, "RUNS_DIR", tmp_path / "runs")


def _tiny_request(media, tmp_path, **kw):
    ref, ctl = media
    over = ["inputs.motion_is_control=true", f"output.dir={(tmp_path / 'out').as_posix()}"]
    kw.setdefault("reference", str(ref))
    kw.setdefault("motion", str(ctl))
    return InferenceRequest(model="tiny", overrides=over + kw.pop("extra", []), **kw)


# --- registry / model interface --------------------------------------------------------------------

def test_registry_names_aliases_and_metadata():
    assert {"wan2.1-vace-1.3b", "wan2.1-vace-tiny-random"} <= set(model_names())
    assert resolve_name("WAN") == resolve_name("wan-vace") == "wan2.1-vace-1.3b"
    spec = get_spec("wan")
    assert spec.runtimes == ("pytorch",) and "cpu" in spec.opt_in_devices
    assert spec.weights.endswith("ec4d2cb062b548996b179d493fdd05340de702a1") and spec.weights_size_gb == 19.04
    assert spec.verification["pytorch/cuda"].startswith("UNVERIFIED")  # never claimed without evidence


def test_unknown_model_is_a_clear_error():
    with pytest.raises(ModelNotFoundError, match="Registered models"):
        get_spec("kling")


def test_duplicate_registration_is_rejected():
    class Clash(MotionModel):
        spec = ModelSpec("x-clash", "1", "f", "t", "d", "l", "c", ("pytorch",), ("cpu",), ("fp32",), (),
                         aliases=("wan",))
    with pytest.raises(ValueError, match="already points to"):
        register_model(Clash)
    assert "x-clash" not in model_names()


def test_model_lifecycle_is_bit_exact_with_phase0_baseline():
    """Same inputs as benchmark/baseline/capture_tiny_vace.py, through registry -> model -> PyTorchRuntime."""
    from common.config import load_config

    rt = get_runtime("pytorch")
    ctx = rt.context(device="cpu", precision="fp32", offload="none")
    model = create_model("tiny", load_config("configs/smoke_tiny_vace.yaml"))
    model.configure(ctx, (32, 32))
    assert not model.loaded
    with pytest.raises(RuntimeError, match="not loaded"):
        model.generate(*synthetic_inputs()[:2])
    model.load(rt, ctx)
    assert model.loaded
    frames, stats = model.generate(*synthetic_inputs()[:2])
    model.unload()
    assert not model.loaded
    digest = frames_digest(frames)
    assert digest["finite"] and digest["shape"] == [5, 32, 32, 3]
    assert digest["sha256"] == BASELINE_SHA
    assert stats["memory"]["device"] == "cpu" and stats["vram_peak_gb"] is None


# --- capability validation -------------------------------------------------------------------------

def test_validate_compatible_and_opt_in():
    rt, ctx = validate(get_spec("tiny"), "pytorch", "cpu", "auto", "auto")
    assert (ctx.device.id, ctx.precision.value, ctx.offload) == ("cpu", "fp32", "none")
    with pytest.raises(DeviceNotSupportedError, match="--allow-cpu"):
        validate(get_spec("wan"), "pytorch", "cpu")
    _, ctx = validate(get_spec("wan"), "pytorch", "cpu", allow_devices=("cpu",))
    assert ctx.device.id == "cpu"


@pytest.mark.parametrize("kw,err", [
    (dict(runtime="onnx"), RuntimeNotAvailableError),
    (dict(runtime="nope"), RuntimeNotAvailableError),
    (dict(device="cuda:7"), DeviceNotSupportedError),
    (dict(device="npu"), DeviceNotSupportedError),
    (dict(precision="int8"), PrecisionNotSupportedError),
    (dict(offload="model"), ConfigError),  # offload on CPU
])
def test_validate_incompatible_combinations(kw, err):
    with pytest.raises(err):
        validate(get_spec("tiny"), **{"runtime": "pytorch", "device": "cpu", **kw})


def test_model_without_runtime_support_and_missing_packages():
    base = get_spec("tiny")
    onnx_only = ModelSpec(**{**base.to_dict(), "runtimes": ("onnx",)})
    with pytest.raises(ModelRuntimeIncompatibleError, match="does not support runtime 'pytorch'"):
        validate(onnx_only, "pytorch", "cpu")
    needs_pkg = ModelSpec(**{**base.to_dict(), "requirements": ("definitely_not_installed_pkg",)})
    with pytest.raises(RuntimeNotAvailableError, match="definitely_not_installed_pkg"):
        validate(needs_pkg, "pytorch", "cpu")
    gpu_only = ModelSpec(**{**base.to_dict(), "devices": ("cuda",)})
    with pytest.raises(DeviceNotSupportedError, match="does not run on cpu"):
        validate(gpu_only, "pytorch", "cpu")


def test_compatibility_matrix_lists_every_runtime():
    rows = {(r.runtime, r.device): r for r in compatibility_matrix(get_spec("wan"))}
    assert rows[("pytorch", "cpu")].status == "opt-in"
    assert rows[("onnx", "*")].status == rows[("tensorrt", "*")].status == "unavailable"


def test_vram_only_shortfall_raises_specific_error():
    vram = ResourceCheck("VRAM free (offload=model)", 3.0, 1.0, "x")
    with pytest.raises(InsufficientVRAMError):
        enforce([vram])
    with pytest.raises(InsufficientResourcesError) as e:
        enforce([vram, ResourceCheck("RAM (pipeline)", 6.0, 1.0, "x")])
    assert not isinstance(e.value, InsufficientVRAMError)
    enforce([vram], override=True)


# --- configuration ------------------------------------------------------------------------------------

def test_runtime_config_precedence(tmp_path):
    rc = tmp_path / "runtime.yaml"
    rc.write_text("runtime: pytorch\ndevice: cpu\nprecision: bf16\noffload: auto\n")
    name, cfg = resolve_run_config(model="tiny", runtime_config=str(rc))
    assert name == "wan2.1-vace-tiny-random" and cfg["runtime"]["precision"] == "bf16"
    _, cfg = resolve_run_config(model="tiny", runtime_config=str(rc), overrides=["runtime.precision=fp16"])
    assert cfg["runtime"]["precision"] == "fp16"  # run config beats runtime.yaml
    _, cfg = resolve_run_config(model="tiny", runtime_config=str(rc), overrides=["runtime.precision=fp16"],
                                runtime_overrides={"precision": "fp32", "device": None})
    assert cfg["runtime"]["precision"] == "fp32" and cfg["runtime"]["device"] == "cpu"  # flags beat everything


def test_legacy_model_dtype_is_migrated(tmp_path):
    _, cfg = resolve_run_config(model="tiny", overrides=["model.dtype=bfloat16", "model.offload=none"])
    assert cfg["runtime"]["precision"] == "bfloat16" and cfg["runtime"]["offload"] == "none"
    assert "dtype" not in cfg["model"]


def test_config_errors(tmp_path):
    with pytest.raises(ConfigError, match="conflicts"):
        resolve_run_config(model="tiny", config="configs/baseline.yaml")
    with pytest.raises(ConfigError, match="Unknown runtime keys"):
        resolve_run_config(model="tiny", overrides=["runtime.gpu=1"])
    bad = tmp_path / "rt.yaml"
    bad.write_text("runtime: pytorch\nthreads: 4\n")
    with pytest.raises(ConfigError, match="Unknown keys"):
        resolve_run_config(model="tiny", runtime_config=str(bad))
    with pytest.raises(ConfigError, match="not found"):
        resolve_run_config(config="configs/missing.yaml")


def test_default_model_uses_baseline_config():
    name, cfg = resolve_run_config()
    assert name == "wan2.1-vace-1.3b" and cfg["model"]["model_id"] == "Wan-AI/Wan2.1-VACE-1.3B-diffusers"


# --- core inference service -------------------------------------------------------------------------

@pytest.mark.parametrize("precision", ["fp32", "bf16", "fp16"])
def test_core_end_to_end_per_precision(media, tmp_path, precision):
    """Precision changed by configuration only; every mode produces a valid, finite video."""
    req = _tiny_request(media, tmp_path, extra=[f"runtime.precision={precision}"],
                        output=str(tmp_path / f"final_{precision}.mp4"))
    res = run_inference(req, echo=lambda _: None)
    assert res.context["precision"] == precision and res.context["device"] == "cpu"
    assert res.stats["output_validation"]["status"] == "PASS"
    assert res.output.is_file() and res.output.name == f"final_{precision}.mp4"
    rec = json.loads(res.record.read_text())
    assert rec["status"] == "success" and rec["runtime"]["precision"] == precision
    assert rec["settings"]["dtype"] == {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}[precision]
    for key in ("generation_time_s", "vram_peak_gb"):  # fields scripts/benchmark.py reads
        assert key in rec["stats"]


def test_core_rejects_bad_inputs_before_running(media, tmp_path):
    with pytest.raises(InvalidInputError, match="Reference image not found"):
        run_inference(_tiny_request(media, tmp_path, reference=str(tmp_path / "nope.png")), echo=lambda _: None)
    with pytest.raises(InvalidInputError, match="Motion video not found"):
        run_inference(_tiny_request(media, tmp_path, motion=str(tmp_path / "nope.mp4")), echo=lambda _: None)
    existing = tmp_path / "exists.mp4"
    existing.write_bytes(b"keep me")
    with pytest.raises(InvalidInputError, match="already exists"):
        run_inference(_tiny_request(media, tmp_path, output=str(existing)), echo=lambda _: None)
    assert existing.read_bytes() == b"keep me"
    assert not (tmp_path / "runs").exists()  # nothing was started


def test_core_refuses_missing_weights_without_download(media, tmp_path, monkeypatch):
    from models.backbones.wan_vace import WanVACEModel

    monkeypatch.setattr(WanVACEModel, "weights_status",
                        lambda self, verify_hashes=False: WeightsStatus(False, 19_040_000_000, "ec4d", {}, "INCOMPLETE"))
    ref, ctl = media
    req = InferenceRequest(model="wan", reference=str(ref), motion=str(ctl), allow_cpu=True,
                           overrides=["inputs.motion_is_control=true"])
    with pytest.raises(ModelWeightsMissingError, match="19.04 GB"):
        run_inference(req, echo=lambda _: None)


def test_second_backbone_plugs_in_without_touching_core(media, tmp_path):
    """A new model only registers itself; core, CLI and runtime are unchanged."""

    @dataclass
    class S:
        width: int
        height: int
        num_frames: int
        fps: int

    class ConstantModel(MotionModel):
        spec = ModelSpec("test-constant", "1", "test", "t", "returns the reference", "n/a", "configs/smoke_tiny_vace.yaml",
                         ("pytorch",), ("cpu", "cuda"), ("fp32",), ("reference-image",))

        def configure(self, ctx, reference_size):
            self.settings = S(32, 32, 5, 16)
            return self.settings

        def weights_status(self, verify_hashes=False):
            return WeightsStatus(True, message="none")

        fetch_weights = weights_status

        def load(self, runtime, ctx):
            self._rt, self._ctx = runtime, ctx
            self._net = runtime.place(torch.nn.Identity(), ctx)

        def generate(self, reference, control):
            res = self._rt.run(lambda: self._net(torch.from_numpy(np.array(reference)).float() / 255), self._ctx)
            return [res.value.numpy()] * len(control), {"generation_time_s": 0.0, "vram_peak_gb": None}

        def unload(self):
            self._net = None

        @property
        def loaded(self):
            return getattr(self, "_net", None) is not None

    register_model(ConstantModel)
    try:
        req = _tiny_request(media, tmp_path)
        req.model = "test-constant"
        res = run_inference(req, echo=lambda _: None)
        assert res.stats["output_validation"]["status"] == "PASS"
        assert json.loads(res.record.read_text())["model"] == "test-constant"
    finally:
        unregister_model("test-constant")


def test_plan_does_no_heavy_work(media, tmp_path):
    plan = plan_inference(_tiny_request(media, tmp_path))
    assert plan.model_name == "wan2.1-vace-tiny-random" and plan.context.runtime == "pytorch"
    assert not (tmp_path / "out").exists()


# --- CLI ------------------------------------------------------------------------------------------

def test_cli_info_json(capsys):
    assert cli(["info", "--model", "wan", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["system"]["devices"][-1]["id"] == "cpu" and data["system"]["pytorch"] == torch.__version__
    assert {r["name"] for r in data["system"]["runtimes"]} == {"pytorch", "onnx", "tensorrt"}
    assert data["model"]["spec"]["name"] == "wan2.1-vace-1.3b"


def test_cli_info_text_mentions_versions(capsys):
    assert cli(["info"]) == 0
    out = capsys.readouterr().out
    for word in ("MOVA", "Python", "PyTorch", "Devices", "Runtimes", "wan2.1-vace-tiny-random"):
        assert word in out


@pytest.mark.parametrize("argv,code", [
    (["info", "--model", "kling"], "model_not_found"),
    (["infer", "--model", "kling"], "model_not_found"),
    (["infer", "--runtime", "tensorrt"], "runtime_not_available"),
    (["infer", "--device", "cuda:3"], "device_not_supported"),
    (["infer", "--model", "tiny", "--precision", "fp8"], "precision_not_supported"),
    (["infer", "--model", "tiny", "--reference", "missing.png"], "invalid_input"),
])
def test_cli_errors_are_structured(argv, code, capsys):
    assert cli(argv) == 2
    assert f"ERROR [{code}]" in capsys.readouterr().err


def test_cli_rejects_invalid_arguments():
    with pytest.raises(SystemExit) as e:
        cli(["infer", "--no-such-flag"])
    assert e.value.code == 2


def test_cli_infer_end_to_end_json(media, tmp_path, capsys):
    ref, ctl = media
    out = tmp_path / "cli.mp4"
    code = cli(["infer", "--model", "tiny", "--runtime", "pytorch", "--device", "cpu", "--precision", "fp32",
                "--reference", str(ref), "--motion", str(ctl), "--output", str(out), "--json",
                "--set", "inputs.motion_is_control=true", "--set", f"output.dir={(tmp_path / 'o').as_posix()}"])
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["runtime"] == {"runtime": "pytorch", "device": "cpu", "device_name": data["runtime"]["device_name"],
                               "backend": "cpu", "precision": "fp32", "offload": "none"}
    assert data["stats"]["output_validation"]["status"] == "PASS" and out.is_file()


def test_cli_forwards_leading_flags(monkeypatch):
    import mova.cli as mcli

    seen = []
    monkeypatch.setattr(mcli, "_forward", lambda cmd: seen.append(cmd) or 0)
    assert cli(["test", "-q", "--co", "tests/x.py"]) == 0
    assert cli(["evaluate", "--help"]) == 0
    assert seen[0][-4:] == ["pytest", "-q", "--co", "tests/x.py"]
    assert seen[1][-2:] == ["evaluate", "--help"] and seen[1][-3].endswith("benchmark.py")


def test_cli_train_without_dataset_is_a_clear_error(capsys):
    assert cli(["train", "--set", "dataset=datasets/manifests/missing.json", "--set", "forbid_identities_from=null"]) == 2
    assert "ERROR [invalid_config]: Training manifest not found" in capsys.readouterr().err


def _mp_models_present() -> bool:
    from preprocessing.mp_models import MODEL_DIR

    return all((MODEL_DIR / f"{n}.task").exists() for n in ("pose_landmarker_full", "face_landmarker", "hand_landmarker"))


@pytest.mark.skipif(not _mp_models_present(), reason="MediaPipe .task models not downloaded")
def test_full_pipeline_with_motion_extraction(tmp_path):
    """reference.png + raw motion.mp4 -> MediaPipe -> pose control -> model -> runtime -> validated output.mp4."""
    skdata = pytest.importorskip("skimage.data")
    import cv2

    img = skdata.astronaut()
    ref = tmp_path / "reference.png"
    Image.fromarray(img).save(ref)
    frames = [cv2.warpAffine(img, cv2.getRotationMatrix2D((256, 256), 6 * np.sin(i / 2), 1.0), (512, 512),
                             borderMode=cv2.BORDER_REFLECT) for i in range(12)]
    motion = write_video(tmp_path / "motion.mp4", frames, fps=16)
    res = run_inference(InferenceRequest(model="tiny", reference=str(ref), motion=str(motion),
                                         output=str(tmp_path / "output.mp4"),
                                         overrides=[f"output.dir={(tmp_path / 'o').as_posix()}"]), echo=lambda _: None)
    summary = json.loads((res.artifacts_dir / "motion" / "summary.json").read_text())
    assert summary["body"]["detection_rate"] >= 0.75
    v = res.stats["output_validation"]
    assert (v["status"], v["num_frames"], v["width"], v["height"], v["fps"]) == ("PASS", 5, 32, 32, 16.0)
