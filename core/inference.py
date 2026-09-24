"""MOVA Core inference service: reference image + motion video -> validated output video.

One implementation shared by `mova infer`, scripts/inference_baseline.py, the benchmark (through that script)
and any future API. It knows nothing about CUDA, PyTorch dtypes or a specific backbone: those come from the
runtime (runtime/) and the registered model (models/).

Order (nothing heavy starts before every cheap check passed):
  config → capability validation → inputs exist → model settings → weights verified (download only if
  allowed) → resource preflight → run record → control video → load → generate → unload → encode + validate.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from common.config import resolve_path
from common.env import detect_hardware, select_profile, summarize
from common.errors import InvalidInputError, ModelWeightsMissingError
from common.experiment import RUNS_DIR, ExperimentRun
from common.logging_utils import get_logger
from common.video_io import read_video, write_video
from evaluation.video import prepare_generated_frames, validate_video
from inference.conditioning import letterbox, prepare_control_frames
from models.registry import create_model, get_spec

from .capabilities import validate
from .config import RUNTIME_CONFIG, resolve_run_config

log = get_logger("mova.core")


@dataclass
class InferenceRequest:
    reference: str | None = None          # None -> config inputs.reference
    motion: str | None = None             # None -> config inputs.motion
    output: str | None = None             # extra copy of output.mp4 at this path
    model: str | None = None              # registry name/alias; None -> config model.name -> default
    config: str | None = None             # None -> the model's default config
    overrides: list[str] = field(default_factory=list)   # --set key=value
    runtime: str | None = None
    device: str | None = None
    precision: str | None = None
    offload: str | None = None
    runtime_config: str = RUNTIME_CONFIG
    allow_download: bool = False
    allow_cpu: bool = False
    verify_hashes: bool = False
    skip_resource_check: bool = False
    overwrite: bool = False
    runs_dir: Path | None = None          # tests only; default experiments/runs


@dataclass
class InferencePlan:
    model_name: str
    config: dict[str, Any]
    runtime: Any
    context: Any


@dataclass
class InferenceResult:
    run_id: str
    record: Path
    artifacts_dir: Path
    output: Path
    stats: dict[str, Any]
    context: dict[str, Any]


def plan_inference(req: InferenceRequest) -> InferencePlan:
    """Resolve config and validate model/runtime/device/precision/offload. Cheap; raises structured errors."""
    name, cfg = resolve_run_config(
        model=req.model, config=req.config, overrides=req.overrides, runtime_config=req.runtime_config,
        runtime_overrides={"runtime": req.runtime, "device": req.device, "precision": req.precision,
                           "offload": req.offload})
    if req.reference:
        cfg["inputs"]["reference"] = req.reference
    if req.motion:
        cfg["inputs"]["motion"] = req.motion
    r = cfg["runtime"]
    rt, ctx = validate(get_spec(name), r["runtime"], r["device"], r["precision"], r["offload"],
                       allow_devices=("cpu",) if req.allow_cpu else ())
    return InferencePlan(name, cfg, rt, ctx)


def build_control(cfg: dict, run_dir: Path) -> list[np.ndarray]:
    inp = cfg["inputs"]
    motion = resolve_path(inp["motion"])
    if inp.get("motion_is_control"):
        return read_video(motion, target_fps=inp.get("motion_fps"))
    from preprocessing.pipeline import ExtractionConfig, extract_motion

    ext_dir = run_dir / "motion"
    extract_motion(motion, ext_dir, ExtractionConfig(target_fps=inp.get("motion_fps"), write_previews=False))
    return read_video(ext_dir / "pose_openpose.mp4")


def side_by_side(*videos: list[np.ndarray]) -> list[np.ndarray]:
    n = min(len(v) for v in videos)
    h = min(v[0].shape[0] for v in videos)
    out = []
    for i in range(n):
        tiles = [np.asarray(Image.fromarray(v[i]).resize((int(v[i].shape[1] * h / v[i].shape[0]), h))) for v in videos]
        out.append(np.concatenate(tiles, axis=1))
    return out


def _input_file(cfg: dict, key: str, what: str) -> Path:
    path = resolve_path(cfg["inputs"][key])
    if not path.is_file():
        raise InvalidInputError(f"{what} not found: {path}")
    return path


def run_inference(req: InferenceRequest, echo: Callable[[str], None] = print) -> InferenceResult:
    plan = plan_inference(req)
    cfg, rt, ctx = plan.config, plan.runtime, plan.context
    hw = detect_hardware(device=ctx.device)
    echo(summarize(hw, select_profile(hw)))
    echo(f"Runtime       : {ctx.runtime} | device {ctx.device.id} ({ctx.device.name}, {ctx.device.backend}) | "
         f"precision {ctx.precision.value} | offload {ctx.offload} | model {plan.model_name}")

    ref_path = _input_file(cfg, "reference", "Reference image")
    _input_file(cfg, "motion", "Motion video")
    if req.output:
        out_copy = Path(req.output).resolve()
        if out_copy.exists() and not req.overwrite:
            raise InvalidInputError(f"Output already exists: {out_copy}", hint="Pass --overwrite or choose another path.")
        if out_copy.suffix.lower() != ".mp4":
            raise InvalidInputError(f"Output must be an .mp4 file, got {out_copy.name}")
    try:
        reference_raw = Image.open(ref_path).convert("RGB")
    except OSError as e:
        raise InvalidInputError(f"Cannot read reference image {ref_path}: {e}") from e

    model = create_model(plan.model_name, cfg)
    s = model.configure(ctx, reference_raw.size)
    weights = model.weights_status(req.verify_hashes)
    echo(f"Model weights : {weights.summary()}")
    spec = model.spec
    if not weights.complete and not req.allow_download:
        raise ModelWeightsMissingError(
            f"Missing {weights.missing_bytes / 1e9:.2f} GB of {spec.weights or spec.name} (license: {spec.license}).",
            hint="Re-run with --allow-download to fetch exactly the files of the pinned manifest.")

    from common.resources import enforce

    output_root = resolve_path(cfg["output"]["dir"])
    checks = model.resource_checks(ctx, weights, output_root)
    if checks:
        echo("Resource preflight (estimates, see common/resources.py):\n  " + "\n  ".join(c.line() for c in checks))
    enforce(checks, override=req.skip_resource_check)

    if not weights.complete:
        weights = model.fetch_weights(req.verify_hashes)

    run = ExperimentRun("baseline", runs_dir=req.runs_dir or RUNS_DIR, artifacts_root=output_root)
    out = run.artifacts_dir
    run.log(dataset=None, resolution=f"{s.width}x{s.height}", frames=s.num_frames, batch=1,
            settings=model.settings_dict(), inputs=cfg["inputs"], hardware=hw.to_dict(), runtime=ctx.to_dict(),
            resource_checks=[c.to_dict() for c in checks], resource_check_skipped=req.skip_resource_check,
            **model.record_fields())
    log.info("Settings: %s", model.settings_dict())

    try:
        control_np = build_control(cfg, out)
        control = prepare_control_frames(control_np, s.width, s.height, s.num_frames,
                                         stride=int(cfg["inputs"].get("frame_stride", 1)))
        reference = letterbox(reference_raw, s.width, s.height)
        reference.save(out / "reference_letterboxed.png")
        write_video(out / "control.mp4", [np.asarray(c) for c in control], s.fps)

        model.load(rt, ctx)
        try:
            frames, stats = model.generate(reference, control)
        finally:
            model.unload()
        frames_u8 = prepare_generated_frames(frames, count=s.num_frames, width=s.width, height=s.height)
        write_video(out / "output.mp4", frames_u8, s.fps)
        stats["output_validation"] = validate_video(
            out / "output.mp4", count=s.num_frames, width=s.width, height=s.height, fps=s.fps)
        if cfg["output"].get("save_side_by_side", True):
            ref_np = np.asarray(reference)
            write_video(out / "side_by_side.mp4",
                        side_by_side([ref_np] * len(frames_u8), [np.asarray(c) for c in control], frames_u8), s.fps)
        final = out / "output.mp4"
        if req.output:
            out_copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(final, out_copy)
            stats["output_copy"] = str(out_copy)
            final = out_copy
    except Exception as e:
        run.finish("failed", error=repr(e))
        log.exception("Inference failed")
        raise
    run.finish("success", vram_peak_gb=stats.get("vram_peak_gb"), stats=stats)
    echo(f"Output        : {final}\nRun record    : {run.dir / 'run.json'}")
    return InferenceResult(run.run_id, run.dir / "run.json", out, final, stats, ctx.to_dict())
