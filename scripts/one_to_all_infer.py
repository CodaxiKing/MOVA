"""Run One-to-All-Animation 1.3B (Apache-2.0, arXiv 2511.22940) on an 8 GB Turing GPU, for comparison with EXP-001.

Runs with the ISOLATED venv (.venv-o2a: diffusers 0.33 / transformers 4.40, as upstream pins), never MOVA's .venv:

  .venv-o2a/Scripts/python scripts/one_to_all_infer.py --reference assets/reference/maya.png \
      --video outputs/motion/dance_o2a.mp4 --out outputs/one_to_all/maya --frames 49

Adaptations to upstream (third_party/One-to-All-Animation @ b3c9d88), none of which changes the model maths:
- Imports: skip opensora package __init__ files (they pull Open-Sora/Hunyuan/CogVideo training code, deepspeed and
  a torchvision API removed in current releases); load only the modules inference needs. dwpose_detector is loaded
  without its import-time DWPose instantiation (unused by the ViTPose path; saves a 351 MB download).
- Weights: VAE and tokenizer from the local Wan2.1-VACE-1.3B snapshot (same SHA-256 as the T2V repo's); UMT5 from
  the same snapshot (bf16) on CPU, once, cached to disk. Nothing is downloaded here.
- Memory/precision: fp16 (Turing has no native bf16), text encoder never on the GPU, model CPU offload.
- A single chunk of --frames (4k+1): upstream's split plan assumes >= 81 frames.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
O2A = ROOT / "third_party/One-to-All-Animation/video-generation"
VACE = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
VACE_REV = "ec4d2cb062b548996b179d493fdd05340de702a1"
O2A_REPO, O2A_REV = "MochunniaN1/One-to-All-1.3b_1", "99dc3796f33b6438d627dc042caa2b43afcbcb50"
NEGATIVE = ("black background, Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, "
            "bad hands, bad teeth, bad eyes, bad limbs, distortion")  # upstream inference_1.3b.py


def _stub_packages() -> None:
    """Register opensora subpackages without running their heavy __init__ files."""
    for name in ("opensora", "opensora.dataset", "opensora.models", "opensora.encoder_variants",
                 "opensora.vae_variants", "opensora.model_variants", "opensora.sample"):
        mod = types.ModuleType(name)
        mod.__path__ = [str(O2A / name.replace(".", "/"))]
        sys.modules[name] = mod
    src = (O2A / "dwpose_utils/dwpose_detector.py").read_text(encoding="utf-8")
    src = src.replace("dwpose_detector_aligned = DWposeDetectorAligned(device=device)", "dwpose_detector_aligned = None")
    src = src.replace("dwpose_detector_raw = DWposeDetectorRaw(device=device)", "dwpose_detector_raw = None")
    pkg = types.ModuleType("dwpose_utils")
    pkg.__path__ = [str(O2A / "dwpose_utils")]
    sys.modules["dwpose_utils"] = pkg
    mod = types.ModuleType("dwpose_utils.dwpose_detector")
    mod.__file__ = str(O2A / "dwpose_utils/dwpose_detector.py")
    mod.__package__ = "dwpose_utils"
    sys.modules["dwpose_utils.dwpose_detector"] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)


def _snapshot(repo: str, rev: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, revision=rev, local_files_only=True))


def text_embeddings(prompt: str, cache: Path):
    """UMT5 on CPU once per (prompt, negative); cached like MOVA's ADR-004."""
    import hashlib

    import torch

    key = hashlib.sha1(f"{VACE_REV}|{prompt}|{NEGATIVE}".encode()).hexdigest()[:16]
    path = cache / f"o2a_prompt_{key}.pt"
    if path.exists():
        d = torch.load(path)
        return d["pe"], d["ne"]
    from transformers import AutoTokenizer, UMT5EncoderModel

    snap = _snapshot(VACE, VACE_REV)
    tok = AutoTokenizer.from_pretrained(snap / "tokenizer")
    enc = UMT5EncoderModel.from_pretrained(snap / "text_encoder", torch_dtype=torch.bfloat16).eval()

    def run(text):  # same as upstream WanPipeline._get_t5_prompt_embeds (512 tokens, zero-padded)
        ids = tok([text], padding="max_length", max_length=512, truncation=True, add_special_tokens=True,
                  return_attention_mask=True, return_tensors="pt")
        with torch.no_grad():
            h = enc(ids.input_ids, ids.attention_mask).last_hidden_state
        n = int(ids.attention_mask.sum())
        out = torch.zeros_like(h)
        out[:, :n] = h[:, :n]
        return out

    pe, ne = run(prompt), run(NEGATIVE)
    del enc
    cache.mkdir(parents=True, exist_ok=True)
    torch.save({"pe": pe, "ne": ne, "prompt": prompt, "negative": NEGATIVE}, path)
    return pe, ne


def resizecrop(image, th, tw):
    """Center crop to the th:tw aspect, then resize (copied from upstream inference_1.3b.py, Apache-2.0)."""
    w, h = image.size
    if h / w > th / tw:
        new_w, new_h = int(w), int(int(w) * th / tw)
    else:
        new_h, new_w = int(h), int(int(h) * tw / th)
    left, top = (w - new_w) / 2, (h - new_h) / 2
    return image.crop((left, top, (w + new_w) / 2, (h + new_h) / 2)).resize((tw, th))


def read_pose_video(path, transform_fn=None):
    """Pose video -> [1, C, T, H, W] in [-1, 1] (copied from upstream inference_1.3b.py, Apache-2.0)."""
    import decord
    import numpy as np
    import torch
    from PIL import Image

    vr = decord.VideoReader(path)
    fps = vr.get_avg_fps() if vr.get_avg_fps() > 0 else 30
    frames = []
    for frame in vr:
        f = frame.asnumpy()
        if transform_fn:
            f = np.array(transform_fn(Image.fromarray(f)))
        frames.append(f)
    t = torch.from_numpy(np.stack(frames)).float().permute(3, 0, 1, 2) / 255.0 * 2 - 1
    return t.unsqueeze(0), int(fps)


def build_pipe(device: str, offload: str = "model"):
    import torch
    from diffusers import AutoencoderKLWan
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from safetensors.torch import load_file

    from opensora.model_variants.wanx_diffusers_src import WanTransformer3DModel_Refextractor_2D_Controlnet_prefix
    from opensora.sample.pipeline_wanx_vhuman_tokenreplace import WanPipeline

    dtype = torch.float16
    snap = _snapshot(VACE, VACE_REV)
    # Same dtype as the transformer, as upstream (pipe.to(dtype)): the pipeline uses vae.dtype for all latents.
    vae = AutoencoderKLWan.from_pretrained(snap / "vae", torch_dtype=dtype)
    model = WanTransformer3DModel_Refextractor_2D_Controlnet_prefix.from_config("configs/wan2.1_t2v_1.3b.json")
    model.set_up_controlnet("configs/wan2.1_t2v_1.3b_controlnet_2.json", dtype)
    model.set_up_refextractor("configs/wan2.1_t2v_1.3b_refextractor_2d_withmask2.json", dtype)
    ckpt = _snapshot(O2A_REPO, O2A_REV)
    state = {}
    for f in sorted(ckpt.glob("*.safetensors")):
        state.update(load_file(str(f), device="cpu"))
    model.load_state_dict(state, strict=True)
    model = model.to(dtype).eval().requires_grad_(False)
    scheduler = FlowMatchEulerDiscreteScheduler(shift=7.0, num_train_timesteps=1000, use_dynamic_shifting=False)
    pipe = WanPipeline(transformer=model, vae=vae, text_encoder=None, tokenizer=None, scheduler=scheduler)
    # model: whole transformer on the GPU during sampling (did not fit at 384x672x49 on 8 GB: WDDM spilled to RAM
    # and step 1 never finished). sequential: layer by layer, slower but bounded VRAM.
    if offload == "sequential":
        pipe.enable_sequential_cpu_offload(device=device)
    else:
        pipe.enable_model_cpu_offload(device=device)
    return pipe


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=49)
    ap.add_argument("--interval", type=int, default=2, help="sample every N source frames")
    ap.add_argument("--max-short", type=int, default=384, help="upstream 1.3B default")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--image-cfg", type=float, default=2.5)
    ap.add_argument("--pose-cfg", type=float, default=1.5)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--align", default="ref", choices=["ref", "pose"])
    ap.add_argument("--offload", default="sequential", choices=["model", "sequential"])
    args = ap.parse_args()
    if (args.frames - 1) % 4:
        ap.error("--frames must be 4k+1")

    reference, video, out = (Path(p).resolve() for p in (args.reference, args.video, args.out))
    out.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(O2A), str(ROOT / "third_party/o2a_shims")]
    os.chdir(O2A)  # upstream code uses ../pretrained_models relative paths
    _stub_packages()

    import numpy as np
    import torch
    from PIL import Image

    t0 = time.perf_counter()
    import infer_preprocess as ip  # noqa: E402

    ip.process_one(str(reference), str(video), args.interval, args.align == "ref", args.align, "", "",
                   True, False, False)
    cache = sorted(Path(ip.cache_base_dir).glob(f"ref_{reference.stem}_driven_{video.stem}_align_{args.align}_*"),
                   key=lambda p: p.stat().st_mtime)[-1]
    t_pre = time.perf_counter() - t0

    pe, ne = text_embeddings(args.prompt, ROOT / "checkpoints/embeds")
    t_txt = time.perf_counter() - t0 - t_pre


    image = Image.open(cache / "image_input.png").convert("RGB")
    h, w = image.height, image.width
    s = args.max_short / min(h, w)
    if s < 1:
        h, w = int(h * s), int(w * s)
    h, w = h // 16 * 16, w // 16 * 16
    tf = lambda im: resizecrop(im, th=h, tw=w)  # noqa: E731
    image = tf(image)
    pose, fps = read_pose_video(str(cache / "pose.mp4"), tf)
    if pose.shape[2] < args.frames:
        raise SystemExit(f"pose video has {pose.shape[2]} frames < {args.frames}")
    pose = pose[:, :, :args.frames]
    pose_img = tf(Image.open(cache / "pose_input.png").convert("RGB"))
    mask = np.array(tf(Image.open(cache / "mask_input.png").convert("L")), dtype=np.float32) / 255.0
    mask_t = torch.from_numpy(mask)[None, None, None]
    src_pose = torch.from_numpy(np.array(pose_img)).float()[None].permute(0, 3, 1, 2)[:, :, None] / 255.0 * 2 - 1

    device = "cuda:0"
    pipe = build_pipe(device, args.offload)
    t_load = time.perf_counter() - t0 - t_pre - t_txt
    torch.cuda.reset_peak_memory_stats()
    t1 = time.perf_counter()
    frames = pipe(image=image, image_mask=mask_t, control_video=pose, prompt_embeds=pe.to(torch.float16),
                  negative_prompt_embeds=ne.to(torch.float16), height=h, width=w, num_frames=args.frames,
                  image_guidance_scale=args.image_cfg, pose_guidance_scale=args.pose_cfg,
                  num_inference_steps=args.steps, generator=torch.Generator(device=device).manual_seed(42),
                  black_image_cfg=True, black_pose_cfg=True, controlnet_conditioning_scale=1.0, return_tensor=True,
                  case1=False, token_replace=False, prev_frames=None, image_pose=src_pose).frames
    t_gen = time.perf_counter() - t1
    arr = ((frames[0].detach().float().cpu() / 2 + 0.5).clamp(0, 1).permute(1, 2, 3, 0).numpy() * 255).astype("uint8")

    import imageio

    out_fps = fps or 15  # pose.mp4 from the preprocessing is already at source_fps / interval
    imageio.mimwrite(out / "output.mp4", list(arr), fps=out_fps, quality=8)
    ctl = ((pose[0] / 2 + 0.5).clamp(0, 1) * 255).byte().permute(1, 2, 3, 0).numpy()
    side = [np.concatenate([np.array(image), c, a], axis=1) for c, a in zip(ctl, arr)]
    imageio.mimwrite(out / "side_by_side.mp4", side, fps=out_fps, quality=8)
    stats = {"model": f"{O2A_REPO}@{O2A_REV}", "code": "ssj9596/One-to-All-Animation@b3c9d88",
             "reference": str(reference), "video": str(video), "frames": args.frames, "height": h, "width": w,
             "fps": out_fps, "steps": args.steps, "image_cfg": args.image_cfg, "pose_cfg": args.pose_cfg,
             "precision": "fp16", "offload": args.offload, "prompt": args.prompt,
             "preprocess_s": round(t_pre, 1), "text_s": round(t_txt, 1), "load_s": round(t_load, 1),
             "generation_s": round(t_gen, 1), "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
             "finite": bool(np.isfinite(frames[0].float().cpu().numpy()).all())}
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    sys.path.insert(0, str(ROOT))
    from common.experiment import ExperimentRun  # rule 4: every model run leaves experiments/runs/<id>/run.json

    run = ExperimentRun("one_to_all", runs_dir=ROOT / "experiments/runs")
    run.log(model=stats["model"], resolution=f"{w}x{h}", frames=args.frames, batch=1, **{k: v for k, v in stats.items()
            if k not in ("model", "frames")})
    run.finish("success", vram_peak_gb=stats["vram_peak_gb"], output=str(out / "output.mp4"))
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
