"""Wan2.1-VACE backbones behind the model interface.

`WanVACEModel` is the Phase 1 baseline (Wan2.1-VACE-1.3B, Diffusers, ADR-002). The numerics stay in
inference/baseline_vace.py; this adapter adds settings resolution, weight verification (ADR-008),
resource preflight and runtime-driven placement.

`WanVACETinyRandomModel` runs the SAME Diffusers code path with a tiny randomly initialised network and no
downloads. It exists to test the whole stack (CLI → core → model → runtime → video) on any machine. Its
output is noise by construction: it proves wiring and numerics, never quality.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from common.config import resolve_path
from common.errors import ModelWeightsMissingError
from common.logging_utils import get_logger
from inference.baseline_vace import DEFAULT_NEGATIVE, BaselineSettings, build_pipeline, encode_prompts_cached, generate
from inference.conditioning import fit_resolution, valid_num_frames

from ..base import ModelSpec, MotionModel, WeightsStatus
from ..registry import register_model

log = get_logger("mova.models.wan_vace")

VACE_REPO = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
VACE_REVISION = "ec4d2cb062b548996b179d493fdd05340de702a1"


def _auto(value, fallback):
    return fallback if value in (None, "auto") else value


@register_model
class WanVACEModel(MotionModel):
    spec = ModelSpec(
        name="wan2.1-vace-1.3b",
        aliases=("wan", "wan-vace"),
        version=f"{VACE_REPO}@{VACE_REVISION[:12]}",
        family="wan2.1-dit-1.3b",
        task="reference image + pose control video -> video",
        description="Phase 1 baseline: Wan2.1-VACE-1.3B via Diffusers WanVACEPipeline (ADR-001/002).",
        license="Apache-2.0",
        default_config="configs/baseline.yaml",
        runtimes=("pytorch",),
        devices=("cuda", "cpu"),
        opt_in_devices={"cpu": "diffusion with the 1.3B model on CPU is impractical (hours); pass --allow-cpu"},
        precisions=("bf16", "fp16", "fp32"),
        capabilities=("pose-video-control", "reference-image", "text-prompt", "model-cpu-offload",
                      "sequential-cpu-offload", "vae-tiling"),
        requirements=("diffusers", "transformers", "accelerate", "sentencepiece", "ftfy"),
        weights=f"{VACE_REPO}@{VACE_REVISION}",
        weights_size_gb=19.04,
        verification={
            "pytorch/cuda": "UNVERIFIED - never run with the real weights (no CUDA machine yet, EXP-001)",
            "pytorch/cpu": "UNVERIFIED with real weights; code path VERIFIED only via wan2.1-vace-tiny-random",
            "onnx": "UNSUPPORTED", "tensorrt": "UNSUPPORTED",
        },
    )

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._pipe = None
        self._embeds: tuple[torch.Tensor, torch.Tensor] | None = None
        self._runtime = None
        self._ctx = None
        self._manifest: dict | None = None
        self._load_stats: dict[str, Any] = {}

    # --- planning ---------------------------------------------------------
    def configure(self, ctx, reference_size: tuple[int, int]) -> BaselineSettings:
        from common.env import detect_hardware, select_profile

        profile = select_profile(detect_hardware(device=ctx.device))  # resolution/frames defaults per VRAM class
        m, g = self.config["model"], self.config["generation"]
        height = int(_auto(g.get("height"), profile.height))
        width = int(_auto(g.get("width"), profile.width))
        if g.get("keep_reference_aspect", True):
            width, height = fit_resolution(reference_size[0], reference_size[1], height * width)
        self.profile = profile
        self.settings = BaselineSettings(
            model_id=m["model_id"],
            revision=m.get("revision"),
            local_files_only=True,
            prompt=g["prompt"],
            negative_prompt=g.get("negative_prompt") or DEFAULT_NEGATIVE,
            height=height,
            width=width,
            num_frames=valid_num_frames(int(_auto(g.get("num_frames"), profile.num_frames))),
            num_inference_steps=int(g["num_inference_steps"]),
            guidance_scale=float(g["guidance_scale"]),
            flow_shift=float(g["flow_shift"]),
            conditioning_scale=float(g["conditioning_scale"]),
            seed=int(g["seed"]),
            dtype=ctx.precision.torch_name,
            offload=ctx.offload,
            vae_tiling=bool(_auto(m.get("vae_tiling"), profile.vae_tiling)),
            fps=int(g["fps"]),
        )
        return self.settings

    def weights_status(self, verify_hashes: bool = False) -> WeightsStatus:
        """Resolve the exact revision and verify every required file (ADR-008), not just model_index.json."""
        from common.hf_utils import load_or_fetch_manifest, verify_cache

        model_id, revision = self.config["model"]["model_id"], self.config["model"].get("revision")
        try:
            manifest = load_or_fetch_manifest(model_id, revision)
        except Exception as e:  # offline without a saved manifest, bad revision, ... (re-raised as structured)
            raise ModelWeightsMissingError(
                f"Cannot resolve file list for {model_id}@{revision or 'main'}: {e}",
                hint="Run once online (scripts/check_model_size.py) to save the manifest.") from e
        if not revision:
            log.warning("model.revision is not pinned; resolved main -> %s. Pin it for reproducibility.",
                        manifest["revision"])
        self._manifest = manifest
        if self.settings is not None:
            self.settings.revision = manifest["revision"]  # always load the exact snapshot that was verified
        cache = verify_cache(manifest, check_hashes=verify_hashes)
        self._cache = cache
        return WeightsStatus(cache.complete, cache.missing_bytes, manifest["revision"], cache.to_dict(),
                             cache.summary())

    def fetch_weights(self, verify_hashes: bool = False) -> WeightsStatus:
        from huggingface_hub import snapshot_download

        manifest = self._manifest or {}
        if not manifest:
            self.weights_status(verify_hashes)
            manifest = self._manifest
        files = [f["path"] for f in manifest["files"]]
        log.info("Downloading %d required files of %s@%s", len(files), manifest["repo_id"], manifest["revision"])
        snapshot_download(manifest["repo_id"], revision=manifest["revision"], allow_patterns=files)
        status = self.weights_status(verify_hashes)
        if not status.complete:
            raise ModelWeightsMissingError(f"Download finished but cache is still incomplete: {status.summary()}")
        return status

    def resource_checks(self, ctx, weights: WeightsStatus, output_dir) -> list:
        from huggingface_hub import constants as hf_constants

        from common.resources import baseline_checks
        from inference.baseline_vace import prompt_cache_path

        s = self.settings
        return baseline_checks(
            missing_download_bytes=0 if weights.complete else weights.missing_bytes,
            hf_cache_dir=hf_constants.HF_HUB_CACHE, output_dir=output_dir,
            need_text_encoder=not prompt_cache_path(s, resolve_path(s.embed_cache_dir)).exists(),
            offload=s.offload, cuda=ctx.device.is_accelerator, device=ctx.device)

    def record_fields(self) -> dict[str, Any]:
        cache = getattr(self, "_cache", None)
        return {"model": self.settings.model_id, "model_name": self.spec.name,
                "model_revision": self.settings.revision,
                "revision_pinned": bool(self.config["model"].get("revision")),
                "model_cache": cache.to_dict() if cache is not None else None,
                "profile": self.profile.to_dict()}

    # --- execution --------------------------------------------------------
    def encode_prompts(self) -> tuple[torch.Tensor, torch.Tensor]:
        return encode_prompts_cached(self.settings, resolve_path(self.settings.embed_cache_dir))

    def build(self):
        return build_pipeline(self.settings)

    def load(self, runtime, ctx) -> None:
        t0 = time.perf_counter()
        self._embeds = self.encode_prompts()
        t1 = time.perf_counter()
        pipe = self.build()
        self._pipe = runtime.place(pipe, ctx)
        self._runtime, self._ctx = runtime, ctx
        self._load_stats = {"prompt_encode_time_s": round(t1 - t0, 1), "load_time_s": round(time.perf_counter() - t1, 1)}

    def generate(self, reference: Image.Image, control: list[Image.Image]) -> tuple[list[Any], dict[str, Any]]:
        if not self.loaded:
            raise RuntimeError("Model is not loaded; call load() first")
        rt, ctx, s = self._runtime, self._ctx, self.settings
        pe, ne = self._embeds
        result = rt.run(lambda: generate(self._pipe, s, reference, control, pe, ne,
                                         generator=rt.generator(s.seed, ctx)), ctx)
        frames, stats = result.value
        stats.update(self._load_stats)
        stats.update(vram_peak_gb=result.memory.peak_allocated_gb,
                     vram_reserved_peak_gb=result.memory.peak_reserved_gb,
                     memory=result.memory.to_dict())
        return frames, stats

    def unload(self) -> None:
        self._pipe = None
        self._embeds = None
        if self._runtime is not None:
            self._runtime.release(self._ctx)
        self._runtime = self._ctx = None

    @property
    def loaded(self) -> bool:
        return self._pipe is not None


def tiny_vace_pipeline(dtype: torch.dtype = torch.float32):
    """Tiny random WanVACE pipeline (same architecture code as the 1.3B, ~0.1 M params, seeded)."""
    from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanVACEPipeline, WanVACETransformer3DModel

    torch.manual_seed(0)
    vae = AutoencoderKLWan(base_dim=3, z_dim=16, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
                           temperal_downsample=[False, True, True])
    transformer = WanVACETransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12, in_channels=16, out_channels=16,
        text_dim=32, freq_dim=32, ffn_dim=32, num_layers=3, cross_attn_norm=True, qk_norm="rms_norm_across_heads",
        rope_max_seq_len=32, vace_layers=[0, 2], vace_in_channels=96,
    )
    if dtype != torch.float32:
        transformer = transformer.to(dtype)
    return WanVACEPipeline(tokenizer=None, text_encoder=None, vae=vae,
                           scheduler=FlowMatchEulerDiscreteScheduler(shift=3.0), transformer=transformer)


@register_model
class WanVACETinyRandomModel(WanVACEModel):
    spec = ModelSpec(
        name="wan2.1-vace-tiny-random",
        aliases=("tiny",),
        version="random-seed0",
        family="wan2.1-dit (tiny, random weights)",
        task="smoke test of the full stack; output is noise",
        description="Same Diffusers WanVACE code path with a tiny random network. No downloads. "
                    "Use to verify CLI/core/runtime/device wiring on any machine.",
        license="Apache-2.0 (code only, no weights)",
        default_config="configs/smoke_tiny_vace.yaml",
        runtimes=("pytorch",),
        devices=("cuda", "cpu"),
        precisions=("fp32", "bf16", "fp16"),
        capabilities=("pose-video-control", "reference-image", "smoke-test-only"),
        requirements=("diffusers",),
        verification={"pytorch/cpu": "VERIFIED (fp32 bit-exact vs benchmark/baseline; bf16/fp16 finite)",
                      "pytorch/cuda": "UNVERIFIED (no CUDA machine yet)"},
    )

    def weights_status(self, verify_hashes: bool = False) -> WeightsStatus:
        return WeightsStatus(True, 0, None, {}, "no weights (random init)")

    def fetch_weights(self, verify_hashes: bool = False) -> WeightsStatus:
        return self.weights_status()

    def resource_checks(self, ctx, weights: WeightsStatus, output_dir) -> list:
        from common.resources import OUTPUT_DISK_GB, ResourceCheck, disk_free_gb

        return [ResourceCheck("disk (outputs)", OUTPUT_DISK_GB, disk_free_gb(output_dir), "videos and run records")]

    def record_fields(self) -> dict[str, Any]:
        return {"model": self.spec.name, "model_name": self.spec.name, "model_revision": None,
                "revision_pinned": False, "model_cache": None, "profile": self.profile.to_dict()}

    def encode_prompts(self) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator().manual_seed(self.settings.seed)
        return torch.randn(1, 8, 32, generator=g), torch.zeros(1, 8, 32)

    def build(self):
        return tiny_vace_pipeline(getattr(torch, self.settings.dtype))

    def settings_dict(self) -> dict[str, Any]:
        return asdict(self.settings) | {"model_id": self.spec.name}
