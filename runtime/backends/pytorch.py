"""Reference runtime: eager PyTorch on CPU or CUDA/ROCm GPUs."""

from __future__ import annotations

from typing import Any, Callable

from ..base import Availability, ExecutionContext, RunResult, Runtime
from ..device import DeviceInfo, DeviceManager, default_device_manager
from ..memory import MemoryManager
from ..precision import PrecisionManager


class PyTorchRuntime(Runtime):
    name = "pytorch"

    def __init__(self, devices: DeviceManager | None = None, precision: PrecisionManager | None = None,
                 memory: MemoryManager | None = None) -> None:
        self.device_manager = devices or default_device_manager()
        self.precision_manager = precision or PrecisionManager()
        self.memory_manager = memory or MemoryManager()

    @classmethod
    def availability(cls) -> Availability:
        try:
            import torch
        except ImportError as e:
            return Availability(False, f"PyTorch is not installed ({e})")
        return Availability(True, "", torch.__version__)

    def devices(self) -> list[DeviceInfo]:
        return self.device_manager.list_devices()

    def context(self, device: str | None = "auto", precision: str | None = "auto", offload: str | None = "auto",
                *, allowed_precisions: tuple[str, ...] | None = None) -> ExecutionContext:
        dev = self.device_manager.resolve(device)
        prec = self.precision_manager.resolve(precision, dev, allowed_precisions)
        off = self.memory_manager.resolve_offload(offload, dev)
        return ExecutionContext(self.name, dev, prec, off)

    def dtype(self, ctx: ExecutionContext) -> Any:
        import torch

        return getattr(torch, ctx.precision.torch_name)

    def place(self, obj: Any, ctx: ExecutionContext) -> Any:
        """Diffusers pipelines get CPU offload hooks bound to ctx.device; anything else is moved with .to()."""
        if ctx.offload == "model":
            obj.enable_model_cpu_offload(device=ctx.device.id)
        elif ctx.offload == "sequential":
            obj.enable_sequential_cpu_offload(device=ctx.device.id)
        else:
            obj = obj.to(ctx.device.id) or obj  # DiffusionPipeline.to returns self; nn.Module.to too
        return obj

    def run(self, fn: Callable[[], Any], ctx: ExecutionContext) -> RunResult:
        import torch

        with self.memory_manager.track(ctx.device) as report, torch.no_grad():
            value = fn()
        return RunResult(value=value, memory=report)

    def generator(self, seed: int, ctx: ExecutionContext) -> Any:
        import torch

        # CPU generator on purpose: the same seed gives the same initial noise on any device.
        return torch.Generator().manual_seed(seed)

    def release(self, ctx: ExecutionContext) -> None:
        self.memory_manager.cleanup(ctx.device)

    def memory_stats(self, ctx: ExecutionContext) -> dict[str, Any]:
        return self.memory_manager.stats(ctx.device)

    def info(self) -> dict[str, Any]:
        d = super().info()
        if d["available"]:
            import torch

            d.update(cuda_build=torch.version.cuda, rocm_build=getattr(torch.version, "hip", None),
                     devices=[x.id for x in self.devices()])
        return d
