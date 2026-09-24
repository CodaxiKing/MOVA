"""Hardware detection and automatic low-VRAM profile selection."""

from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

GIB = 1024**3


@dataclass
class HardwareInfo:
    python: str
    platform: str
    torch_version: str
    torch_cuda_build: str | None
    cuda_available: bool
    device: str
    gpu_name: str | None = None
    vram_total_gb: float | None = None
    vram_free_gb: float | None = None
    compute_capability: str | None = None
    bf16_supported: bool = False
    ram_total_gb: float | None = None
    ram_available_gb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeProfile:
    """Settings chosen automatically from the detected hardware."""

    name: str
    device: str
    dtype: str
    height: int
    width: int
    num_frames: int
    offload: str  # "none" | "model" | "sequential"
    vae_tiling: bool
    notes: list[str] = field(default_factory=list)

    @property
    def torch_dtype(self) -> torch.dtype:
        return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[self.dtype]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _system_ram() -> tuple[float | None, float | None]:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return round(vm.total / GIB, 2), round(vm.available / GIB, 2)
    except ImportError:
        return None, None


def detect_hardware() -> HardwareInfo:
    ram_total, ram_avail = _system_ram()
    info = HardwareInfo(
        python=sys.version.split()[0],
        platform=platform.platform(),
        torch_version=torch.__version__,
        torch_cuda_build=torch.version.cuda,
        cuda_available=torch.cuda.is_available(),
        device="cuda" if torch.cuda.is_available() else "cpu",
        ram_total_gb=ram_total,
        ram_available_gb=ram_avail,
    )
    if info.cuda_available:
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        info.gpu_name = props.name
        info.vram_total_gb = round(total / GIB, 2)
        info.vram_free_gb = round(free / GIB, 2)
        info.compute_capability = f"{props.major}.{props.minor}"
        info.bf16_supported = bool(torch.cuda.is_bf16_supported())
    return info


def select_profile(hw: HardwareInfo) -> RuntimeProfile:
    """Pick resolution/frames/offload for the Wan2.1 1.3B family given the hardware.

    Thresholds are conservative starting points; measured numbers go to docs/experiments.
    """
    if not hw.cuda_available:
        return RuntimeProfile(
            name="cpu",
            device="cpu",
            dtype="float32",
            height=256,
            width=256,
            num_frames=9,
            offload="none",
            vae_tiling=True,
            notes=["No CUDA GPU: diffusion inference is impractical. Use only for preprocessing and tests."],
        )

    dtype = "bfloat16" if hw.bf16_supported else "float16"
    vram = hw.vram_total_gb or 0.0
    if vram < 6.5:
        return RuntimeProfile("cuda-lt6gb", "cuda", dtype, 256, 256, 9, "sequential", True,
                              ["Very low VRAM: sequential offload, expect slow inference."])
    if vram < 10:
        return RuntimeProfile("cuda-8gb", "cuda", dtype, 256, 256, 17, "model", True,
                              ["8 GB class (e.g. RTX 3060 8GB): model offload + VAE tiling."])
    if vram < 20:
        return RuntimeProfile("cuda-12-16gb", "cuda", dtype, 480, 480, 33, "model", True)
    return RuntimeProfile("cuda-24gb+", "cuda", dtype, 480, 832, 81, "none", False)


def summarize(hw: HardwareInfo, profile: RuntimeProfile) -> str:
    lines = [
        f"Python        : {hw.python}",
        f"Platform      : {hw.platform}",
        f"PyTorch       : {hw.torch_version} (CUDA build: {hw.torch_cuda_build})",
        f"CUDA available: {hw.cuda_available}",
    ]
    if hw.cuda_available:
        lines += [
            f"GPU           : {hw.gpu_name} (cc {hw.compute_capability})",
            f"VRAM          : {hw.vram_free_gb} GB free / {hw.vram_total_gb} GB total",
            f"BF16          : {hw.bf16_supported}",
        ]
    if hw.ram_total_gb is not None:
        lines.append(f"System RAM    : {hw.ram_available_gb} GB available / {hw.ram_total_gb} GB total")
    lines += [
        f"Profile       : {profile.name} -> {profile.width}x{profile.height}, {profile.num_frames} frames, "
        f"dtype={profile.dtype}, offload={profile.offload}, vae_tiling={profile.vae_tiling}",
    ]
    lines += [f"Note          : {n}" for n in profile.notes]
    return "\n".join(lines)
