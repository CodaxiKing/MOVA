"""Phase 1 baseline: reference image + motion video -> generated video with Wan2.1-VACE-1.3B.

Safety gates:
  * Refuses to download model weights unless --allow-download is given (prints size + license first).
  * Refuses to run diffusion on CPU unless --allow-cpu is given.

Usage:
  python scripts/check_model_size.py Wan-AI/Wan2.1-VACE-1.3B-diffusers
  python scripts/inference_baseline.py --config configs/baseline.yaml --allow-download
  python scripts/inference_baseline.py --set generation.num_frames=9 --set generation.height=256
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from common.config import load_config, resolve_path  # noqa: E402
from common.env import detect_hardware, select_profile, summarize  # noqa: E402
from common.experiment import ExperimentRun  # noqa: E402
from common.logging_utils import get_logger  # noqa: E402
from common.video_io import read_video, write_video  # noqa: E402
from evaluation.video import prepare_generated_frames, validate_video  # noqa: E402
from inference.conditioning import fit_resolution, letterbox, prepare_control_frames, valid_num_frames  # noqa: E402

log = get_logger("mova.baseline")
LICENSES = {"Wan-AI/Wan2.1-VACE-1.3B-diffusers": "Apache-2.0"}


def resolve_settings(cfg: dict, profile, ref_size: tuple[int, int]):
    from inference.baseline_vace import DEFAULT_NEGATIVE, BaselineSettings

    m, g = cfg["model"], cfg["generation"]
    auto = lambda v, fallback: fallback if v in (None, "auto") else v  # noqa: E731
    height = int(auto(g["height"], profile.height))
    width = int(auto(g["width"], profile.width))
    if g.get("keep_reference_aspect", True):
        width, height = fit_resolution(ref_size[0], ref_size[1], height * width)
    return BaselineSettings(
        model_id=m["model_id"],
        prompt=g["prompt"],
        negative_prompt=g.get("negative_prompt") or DEFAULT_NEGATIVE,
        height=height,
        width=width,
        num_frames=valid_num_frames(int(auto(g["num_frames"], profile.num_frames))),
        num_inference_steps=int(g["num_inference_steps"]),
        guidance_scale=float(g["guidance_scale"]),
        flow_shift=float(g["flow_shift"]),
        conditioning_scale=float(g["conditioning_scale"]),
        seed=int(g["seed"]),
        dtype=auto(m["dtype"], profile.dtype),
        offload=auto(m["offload"], profile.offload),
        vae_tiling=bool(auto(m["vae_tiling"], profile.vae_tiling)),
        fps=int(g["fps"]),
    )


def build_control(cfg: dict, run_dir: Path) -> list[np.ndarray]:
    inp = cfg["inputs"]
    motion = resolve_path(inp["motion"])
    if not motion.exists():
        raise SystemExit(f"Motion video not found: {motion}")
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--reference", default=None)
    ap.add_argument("--motion", default=None)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    if args.reference:
        cfg["inputs"]["reference"] = args.reference
    if args.motion:
        cfg["inputs"]["motion"] = args.motion

    hw = detect_hardware()
    profile = select_profile(hw)
    print(summarize(hw, profile))
    if not hw.cuda_available and not args.allow_cpu:
        raise SystemExit("No CUDA GPU detected. Diffusion inference on CPU is impractical; run on the RTX 3060 "
                         "machine, or pass --allow-cpu to force (expect hours).")

    ref_path = resolve_path(cfg["inputs"]["reference"])
    if not ref_path.exists():
        raise SystemExit(f"Reference image not found: {ref_path}")
    reference_raw = Image.open(ref_path).convert("RGB")

    from common.hf_utils import is_cached, repo_size

    model_id = cfg["model"]["model_id"]
    if not is_cached(model_id) and not args.allow_download:
        sizes = repo_size(model_id)
        raise SystemExit(
            f"Model {model_id} is not cached. Download size: {sizes['TOTAL']} GB "
            f"(breakdown: {sizes}); license: {LICENSES.get(model_id, 'check model card')}. "
            "Re-run with --allow-download after confirming disk space (~25 GB recommended).")

    s = resolve_settings(cfg, profile, reference_raw.size)
    run = ExperimentRun("baseline", artifacts_root=resolve_path(cfg["output"]["dir"]))
    out = run.artifacts_dir
    run.log(model=s.model_id, dataset=None, resolution=f"{s.width}x{s.height}", frames=s.num_frames, batch=1,
            settings=vars(s), inputs=cfg["inputs"], hardware=hw.to_dict(), profile=profile.to_dict())
    log.info("Settings: %s", vars(s))

    try:
        control_np = build_control(cfg, out)
        control = prepare_control_frames(control_np, s.width, s.height, s.num_frames,
                                         stride=int(cfg["inputs"].get("frame_stride", 1)))
        reference = letterbox(reference_raw, s.width, s.height)
        reference.save(out / "reference_letterboxed.png")
        write_video(out / "control.mp4", [np.asarray(c) for c in control], s.fps)

        from inference.baseline_vace import run_baseline

        frames, stats = run_baseline(s, reference, control, resolve_path(s.embed_cache_dir))
        frames_u8 = prepare_generated_frames(frames, count=s.num_frames, width=s.width, height=s.height)
        write_video(out / "output.mp4", frames_u8, s.fps)
        stats["output_validation"] = validate_video(
            out / "output.mp4", count=s.num_frames, width=s.width, height=s.height, fps=s.fps)
        if cfg["output"].get("save_side_by_side", True):
            ref_np = np.asarray(reference)
            write_video(out / "side_by_side.mp4",
                        side_by_side([ref_np] * len(frames_u8), [np.asarray(c) for c in control], frames_u8), s.fps)
    except Exception as e:
        run.finish("failed", error=repr(e))
        log.exception("Baseline failed")
        raise
    run.finish("success", vram_peak_gb=stats.get("vram_peak_gb"), stats=stats)
    log.info("Done: %s | record: %s | stats=%s", out, run.dir / "run.json", stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
