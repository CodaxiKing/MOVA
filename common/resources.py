"""Resource preflight: refuse to start work that is known not to fit in disk, RAM or VRAM.

Requirements are ESTIMATES derived from file sizes (documented next to each constant); they must be
recalibrated with the measured peaks of EXP-001. Every check reports what it assumed.
"""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

GB = 1e9

# UMT5-XXL encoder: 11.36 GB of bf16 safetensors (HF API) + tokenizer/activations/Python overhead.
TEXT_ENCODER_RAM_GB = 13.0
# VACE-1.3B transformer (7.15 GB fp32 on disk -> ~3.6 GB bf16) + fp32 VAE (0.51 GB) held in RAM while offloaded.
PIPELINE_RAM_GB = {"model": 6.0, "sequential": 6.0, "none": 2.0}
# Minimum free VRAM to start; the real peak is measured and recorded in run.json.
PIPELINE_VRAM_GB = {"model": 3.0, "sequential": 1.5, "none": 6.0}
DOWNLOAD_MARGIN_GB = 2.0
OUTPUT_DISK_GB = 1.0


class InsufficientResources(RuntimeError):
    pass


@dataclass
class ResourceCheck:
    name: str
    required_gb: float
    available_gb: float | None
    reason: str

    @property
    def ok(self) -> bool:
        return self.available_gb is None or self.available_gb >= self.required_gb

    def to_dict(self) -> dict:
        return {**asdict(self), "ok": self.ok}

    def line(self) -> str:
        avail = "unknown" if self.available_gb is None else f"{self.available_gb:.2f} GB"
        return f"[{'OK ' if self.ok else 'NO '}] {self.name}: need {self.required_gb:.2f} GB, have {avail} - {self.reason}"


def disk_free_gb(path: str | Path) -> float:
    p = Path(path).resolve()
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free / GB


def ram_available_gb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    return psutil.virtual_memory().available / GB


def vram_free_gb() -> float | None:
    import torch

    if not torch.cuda.is_available():
        return None
    free, _ = torch.cuda.mem_get_info(0)
    return free / GB


def baseline_checks(*, missing_download_bytes: int, hf_cache_dir: str | Path, output_dir: str | Path,
                    need_text_encoder: bool, offload: str, cuda: bool,
                    ram_gb: float | None = None, vram_gb: float | None = None,
                    cache_disk_gb: float | None = None, output_disk_gb: float | None = None) -> list[ResourceCheck]:
    """Measured values can be injected (tests); otherwise they are read from the machine."""
    checks: list[ResourceCheck] = []
    if missing_download_bytes > 0:
        need = missing_download_bytes / GB * 1.05 + DOWNLOAD_MARGIN_GB
        checks.append(ResourceCheck("disk (HF cache)", need,
                                    cache_disk_gb if cache_disk_gb is not None else disk_free_gb(hf_cache_dir),
                                    f"download of {missing_download_bytes / GB:.2f} GB + 5% + {DOWNLOAD_MARGIN_GB} GB margin"))
    checks.append(ResourceCheck("disk (outputs)", OUTPUT_DISK_GB,
                                output_disk_gb if output_disk_gb is not None else disk_free_gb(output_dir),
                                "videos, control frames and run records"))
    ram = ram_gb if ram_gb is not None else ram_available_gb()
    if need_text_encoder:
        checks.append(ResourceCheck("RAM (text encoder)", TEXT_ENCODER_RAM_GB, ram,
                                    "UMT5-XXL bf16 on CPU; only needed when the prompt is not cached yet"))
    checks.append(ResourceCheck(f"RAM (pipeline, offload={offload})", PIPELINE_RAM_GB[offload], ram,
                                "transformer + VAE kept in CPU memory between GPU uses"))
    if cuda:
        checks.append(ResourceCheck(f"VRAM free (offload={offload})", PIPELINE_VRAM_GB[offload],
                                    vram_gb if vram_gb is not None else vram_free_gb(),
                                    "minimum to start; peak is measured during the run"))
    return checks


def enforce(checks: list[ResourceCheck], override: bool = False) -> None:
    failed = [c for c in checks if not c.ok]
    if failed and not override:
        raise InsufficientResources("Insufficient resources; nothing was started:\n  " +
                                    "\n  ".join(c.line() for c in failed) +
                                    "\nFree resources, lower frames/resolution, or pass --skip-resource-check "
                                    "if you know the estimate is wrong (record why).")
