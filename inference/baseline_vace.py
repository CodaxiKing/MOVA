"""Baseline: Wan2.1-VACE-1.3B (Diffusers) — reference image + pose control video -> video.

Memory strategy for 8 GB GPUs (ADR-004):
  1. UMT5 text encoder runs once on CPU; prompt embeddings are cached to disk.
  2. The generation pipeline is then loaded WITHOUT the text encoder.
  3. Model CPU offload (or sequential offload) + VAE tiling.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from common.logging_utils import get_logger

log = get_logger("mova.baseline")

DEFAULT_NEGATIVE = (
    "Bright tones, overexposed, static, blurred details, subtitles, worst quality, low quality, JPEG compression "
    "residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, many people in the background"
)


@dataclass
class BaselineSettings:
    model_id: str = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
    prompt: str = "A person dancing, full body, clean background, high quality"
    negative_prompt: str = DEFAULT_NEGATIVE
    height: int = 256
    width: int = 256
    num_frames: int = 17
    num_inference_steps: int = 30
    guidance_scale: float = 5.0
    flow_shift: float = 3.0
    conditioning_scale: float = 1.0
    seed: int = 42
    dtype: str = "bfloat16"
    offload: str = "model"  # none | model | sequential
    vae_tiling: bool = True
    fps: int = 16
    embed_cache_dir: str = "checkpoints/embeds"


def _dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def encode_prompts_cached(s: BaselineSettings, cache_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode prompt/negative on CPU once; reuse from disk afterwards."""
    key = hashlib.sha1(f"{s.model_id}|{s.prompt}|{s.negative_prompt}".encode()).hexdigest()[:16]
    path = cache_dir / f"prompt_{key}.pt"
    if path.exists():
        d = torch.load(path, map_location="cpu")
        log.info("Loaded cached prompt embeddings %s", path.name)
        return d["prompt_embeds"], d["negative_prompt_embeds"]

    from transformers import AutoTokenizer, UMT5EncoderModel

    log.info("Encoding prompts on CPU with UMT5 (one-time, needs ~12 GB system RAM)...")
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(s.model_id, subfolder="tokenizer")
    enc = UMT5EncoderModel.from_pretrained(s.model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    enc.eval()

    def _enc(text: str) -> torch.Tensor:
        ids = tok([text], padding="max_length", max_length=512, truncation=True, add_special_tokens=True,
                  return_attention_mask=True, return_tensors="pt")
        with torch.no_grad():
            h = enc(ids.input_ids, ids.attention_mask).last_hidden_state
        n = int(ids.attention_mask.sum())
        out = torch.zeros_like(h)
        out[:, :n] = h[:, :n]  # Wan zero-pads beyond the real sequence length
        return out

    pe, ne = _enc(s.prompt), _enc(s.negative_prompt)
    del enc
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"prompt_embeds": pe, "negative_prompt_embeds": ne, "prompt": s.prompt,
                "negative_prompt": s.negative_prompt}, path)
    log.info("Prompt embeddings cached to %s (%.1fs)", path, time.perf_counter() - t0)
    return pe, ne


def load_pipeline(s: BaselineSettings):
    from diffusers import AutoencoderKLWan, WanVACEPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    dtype = _dtype(s.dtype)
    vae = AutoencoderKLWan.from_pretrained(s.model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanVACEPipeline.from_pretrained(s.model_id, vae=vae, text_encoder=None, tokenizer=None, torch_dtype=dtype)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=s.flow_shift)
    if s.vae_tiling:
        pipe.vae.enable_tiling()
    if s.offload == "model":
        pipe.enable_model_cpu_offload()
    elif s.offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cuda")
    return pipe


def run_baseline(s: BaselineSettings, reference: Image.Image, control: list[Image.Image],
                 cache_dir: Path) -> tuple[list[Any], dict[str, Any]]:
    pe, ne = encode_prompts_cached(s, cache_dir)
    t_load = time.perf_counter()
    pipe = load_pipeline(s)
    load_s = time.perf_counter() - t_load
    frames, stats = generate(pipe, s, reference, control, pe, ne)
    return frames, {"load_time_s": round(load_s, 1), **stats}


def generate(pipe, s: BaselineSettings, reference: Image.Image, control: list[Image.Image],
             prompt_embeds: torch.Tensor, negative_prompt_embeds: torch.Tensor) -> tuple[list[Any], dict[str, Any]]:
    if len(control) != s.num_frames:
        raise ValueError(f"control has {len(control)} frames, expected {s.num_frames}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    mask = [Image.new("L", (s.width, s.height), 255)] * s.num_frames  # white = generate everywhere
    dtype = _dtype(s.dtype)
    pe, ne = prompt_embeds, negative_prompt_embeds
    t0 = time.perf_counter()
    out = pipe(
        video=control,
        mask=mask,
        reference_images=[reference],
        conditioning_scale=s.conditioning_scale,
        prompt_embeds=pe.to(dtype),
        negative_prompt_embeds=ne.to(dtype),
        height=s.height,
        width=s.width,
        num_frames=s.num_frames,
        num_inference_steps=s.num_inference_steps,
        guidance_scale=s.guidance_scale,
        generator=torch.Generator().manual_seed(s.seed),
    ).frames[0]
    gen_s = time.perf_counter() - t0
    stats = {
        "generation_time_s": round(gen_s, 1),
        "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2) if torch.cuda.is_available() else None,
        "vram_reserved_peak_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2) if torch.cuda.is_available() else None,
    }
    return out, stats
