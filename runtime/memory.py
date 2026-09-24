"""Memory strategy and accounting for the PyTorch runtime.

Implemented: offload policy per device (none / model / sequential, applied by the runtime), memory
statistics, peak tracking and cleanup. Not implemented (no fake switches): quantisation, attention
optimisations, VAE tiling policy (model-specific, stays in the model settings).
"""

from __future__ import annotations

import gc
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from common.errors import ConfigError

from .device import GIB, DeviceInfo

OFFLOAD_MODES = ("none", "model", "sequential")

# VRAM thresholds for the automatic offload choice (initial guesses for Wan2.1 1.3B, ADR-004;
# recalibrate with EXP-001). Single source: common.env.select_profile uses recommend_offload().
SEQUENTIAL_BELOW_GB = 6.5
MODEL_OFFLOAD_BELOW_GB = 20.0


def recommend_offload(vram_total_gb: float | None) -> str:
    vram = vram_total_gb or 0.0
    if vram < SEQUENTIAL_BELOW_GB:
        return "sequential"
    if vram < MODEL_OFFLOAD_BELOW_GB:
        return "model"
    return "none"


@dataclass
class MemoryReport:
    device: str
    before: dict[str, Any]
    after: dict[str, Any] = field(default_factory=dict)
    peak_allocated_gb: float | None = None
    peak_reserved_gb: float | None = None
    wall_time_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"device": self.device, "before": self.before, "after": self.after,
                "peak_allocated_gb": self.peak_allocated_gb, "peak_reserved_gb": self.peak_reserved_gb,
                "wall_time_s": self.wall_time_s}


class MemoryManager:
    def offload_modes(self, device: DeviceInfo) -> tuple[str, ...]:
        # CPU offload moves weights between host RAM and an accelerator: meaningless when running on CPU.
        return OFFLOAD_MODES if device.is_accelerator else ("none",)

    def resolve_offload(self, requested: str | None, device: DeviceInfo) -> str:
        if requested in (None, "auto"):
            return recommend_offload(device.total_memory_gb) if device.is_accelerator else "none"
        if requested not in OFFLOAD_MODES:
            raise ConfigError(f"Unknown offload mode {requested!r}.", hint=f"Use auto or one of {OFFLOAD_MODES}.")
        if requested not in self.offload_modes(device):
            raise ConfigError(f"Offload {requested!r} needs an accelerator; device is {device.id}.",
                              hint="Use offload none (or auto) on CPU.")
        return requested

    def stats(self, device: DeviceInfo) -> dict[str, Any]:
        """Current usage. Accelerator: torch allocator + driver free/total. CPU: process RSS + system RAM."""
        if device.is_accelerator:
            import torch

            i = device.index or 0
            free, total = torch.cuda.mem_get_info(i)
            return {"allocated_gb": round(torch.cuda.memory_allocated(i) / GIB, 3),
                    "reserved_gb": round(torch.cuda.memory_reserved(i) / GIB, 3),
                    "free_gb": round(free / GIB, 3), "total_gb": round(total / GIB, 3)}
        try:
            import psutil

            vm = psutil.virtual_memory()
            return {"process_rss_gb": round(psutil.Process().memory_info().rss / GIB, 3),
                    "system_available_gb": round(vm.available / GIB, 3), "system_total_gb": round(vm.total / GIB, 3)}
        except ImportError:
            return {}

    def reset_peak(self, device: DeviceInfo) -> None:
        if device.is_accelerator:
            import torch

            torch.cuda.reset_peak_memory_stats(device.index or 0)

    def peak(self, device: DeviceInfo) -> tuple[float | None, float | None]:
        """(peak allocated, peak reserved) in GB since the last reset; None on CPU (not measured)."""
        if not device.is_accelerator:
            return None, None
        import torch

        i = device.index or 0
        return (round(torch.cuda.max_memory_allocated(i) / GIB, 2), round(torch.cuda.max_memory_reserved(i) / GIB, 2))

    def cleanup(self, device: DeviceInfo) -> None:
        gc.collect()
        if device.is_accelerator:
            import torch

            torch.cuda.empty_cache()

    @contextmanager
    def track(self, device: DeviceInfo) -> Iterator[MemoryReport]:
        report = MemoryReport(device=device.id, before=self.stats(device))
        self.reset_peak(device)
        t0 = time.perf_counter()
        try:
            yield report
        finally:
            report.wall_time_s = round(time.perf_counter() - t0, 3)
            report.peak_allocated_gb, report.peak_reserved_gb = self.peak(device)
            report.after = self.stats(device)
