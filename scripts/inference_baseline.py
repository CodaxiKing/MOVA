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
        revision=m.get("revision"),
        local_files_only=True,
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


def check_model_cache(model_id: str, revision: str | None, check_hashes: bool):
    """Resolve the exact revision and verify every required file, not just model_index.json."""
    from common.hf_utils import load_or_fetch_manifest, verify_cache

    try:
        manifest = load_or_fetch_manifest(model_id, revision)
    except Exception as e:  # offline without a saved manifest, bad revision, ...
        raise SystemExit(f"Cannot resolve file list for {model_id}@{revision or 'main'}: {e}\n"
                         "Run once online (scripts/check_model_size.py) to save the manifest.")
    if not revision:
        log.warning("model.revision is not pinned; resolved main -> %s. Pin it in the config for reproducibility.",
                    manifest["revision"])
    cache = verify_cache(manifest, check_hashes=check_hashes)
    print(f"Model cache: {cache.summary()}")
    return manifest, cache


def download_missing(manifest: dict, check_hashes: bool):
    from huggingface_hub import snapshot_download

    from common.hf_utils import verify_cache

    files = [f["path"] for f in manifest["files"]]
    log.info("Downloading %d required files of %s@%s", len(files), manifest["repo_id"], manifest["revision"])
    snapshot_download(manifest["repo_id"], revision=manifest["revision"], allow_patterns=files)
    cache = verify_cache(manifest, check_hashes=check_hashes)
    if not cache.complete:
        raise SystemExit(f"Download finished but cache is still incomplete: {cache.summary()}")
    return cache


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
    ap.add_argument("--verify-hashes", action="store_true", help="also SHA-256 every cached weight file (slow)")
    ap.add_argument("--skip-resource-check", action="store_true", help="start even if the preflight says no")
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

    model_id = cfg["model"]["model_id"]
    manifest, cache = check_model_cache(model_id, cfg["model"].get("revision"), args.verify_hashes)
    s = resolve_settings(cfg, profile, reference_raw.size)
    s.revision = manifest["revision"]  # always load the exact snapshot that was verified
    if not cache.complete and not args.allow_download:
        raise SystemExit(
            f"Missing {cache.missing_bytes / 1e9:.2f} GB of {model_id}@{s.revision[:12]} "
            f"(license: {LICENSES.get(model_id, 'check model card')}). "
            "Re-run with --allow-download to fetch exactly these files.")

    from huggingface_hub import constants as hf_constants

    from common.resources import InsufficientResources, baseline_checks, enforce
    from inference.baseline_vace import prompt_cache_path

    checks = baseline_checks(
        missing_download_bytes=0 if cache.complete else cache.missing_bytes,
        hf_cache_dir=hf_constants.HF_HUB_CACHE, output_dir=resolve_path(cfg["output"]["dir"]),
        need_text_encoder=not prompt_cache_path(s, resolve_path(s.embed_cache_dir)).exists(),
        offload=s.offload, cuda=hw.cuda_available)
    print("Resource preflight (estimates, see common/resources.py):\n  " + "\n  ".join(c.line() for c in checks))
    try:
        enforce(checks, override=args.skip_resource_check)
    except InsufficientResources as e:
        raise SystemExit(str(e))

    if not cache.complete:
        cache = download_missing(manifest, args.verify_hashes)

    run = ExperimentRun("baseline", artifacts_root=resolve_path(cfg["output"]["dir"]))
    out = run.artifacts_dir
    run.log(model=s.model_id, dataset=None, resolution=f"{s.width}x{s.height}", frames=s.num_frames, batch=1,
            settings=vars(s), inputs=cfg["inputs"], hardware=hw.to_dict(), profile=profile.to_dict(),
            model_revision=s.revision, revision_pinned=bool(cfg["model"].get("revision")),
            model_cache=cache.to_dict(), resource_checks=[c.to_dict() for c in checks],
            resource_check_skipped=args.skip_resource_check)
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
