"""Device discovery and selection. The only module (besides diagnostics) that queries torch.cuda.

Devices are identified by PyTorch device strings (`cpu`, `cuda:0`, `cuda:1`, …). ROCm builds of PyTorch
expose AMD GPUs through the same `cuda` device type; they are reported with backend "rocm" (UNVERIFIED:
no AMD hardware has been tested). Apple MPS and other accelerators are not supported yet and are rejected
explicitly instead of silently falling back.
"""

from __future__ import annotations

import platform
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from common.errors import DeviceNotSupportedError

GIB = 1024**3
_DEVICE_RE = re.compile(r"^(cpu|cuda)(?::(\d+))?$")


@dataclass(frozen=True)
class DeviceInfo:
    id: str                     # torch device string, e.g. "cuda:0"
    type: str                   # "cpu" | "cuda"
    backend: str                # "cpu" | "cuda" | "rocm"
    name: str
    index: int | None = None
    total_memory_gb: float | None = None     # VRAM for accelerators, system RAM for cpu
    compute_capability: str | None = None
    bf16: bool = False
    fp16: bool = False
    tensor_cores: bool | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_accelerator(self) -> bool:
        return self.type != "cpu"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DeviceProbe(Protocol):
    """What DeviceManager needs from the framework. Injected in tests to simulate hardware."""

    def accelerator_backend(self) -> str | None: ...
    def accelerator_count(self) -> int: ...
    def accelerator(self, index: int) -> DeviceInfo: ...
    def mem_get_info(self, index: int) -> tuple[int, int]: ...
    def cpu(self) -> DeviceInfo: ...


class TorchDeviceProbe:
    # CPU fp16/bf16: ran end-to-end with the tiny WanVACE pipeline on torch 2.14 (2026-09-24); fp32 stays the
    # CPU default. Not measured with the real 1.3B model.

    def __init__(self) -> None:
        import torch

        self._torch = torch

    def accelerator_backend(self) -> str | None:
        torch = self._torch
        if not torch.cuda.is_available():
            return None
        return "rocm" if getattr(torch.version, "hip", None) else "cuda"

    def accelerator_count(self) -> int:
        return self._torch.cuda.device_count() if self.accelerator_backend() else 0

    def accelerator(self, index: int) -> DeviceInfo:
        torch = self._torch
        props = torch.cuda.get_device_properties(index)
        backend = self.accelerator_backend() or "cuda"
        cc = f"{props.major}.{props.minor}"
        with torch.cuda.device(index):
            # Native only: the default also counts emulation, which says True on Turing (cc 7.5) where bf16 runs
            # ~3x slower than fp16 (EXP-001, RTX 2060 SUPER: 130 s vs 44.7 s).
            bf16 = bool(torch.cuda.is_bf16_supported(including_emulation=False))
        notes = ("ROCm device: UNVERIFIED in MOVA",) if backend == "rocm" else ()
        return DeviceInfo(
            id=f"cuda:{index}", type="cuda", backend=backend, name=props.name, index=index,
            total_memory_gb=round(props.total_memory / GIB, 2), compute_capability=cc, bf16=bf16, fp16=True,
            tensor_cores=(props.major >= 7) if backend == "cuda" else None, notes=notes,
        )

    def mem_get_info(self, index: int) -> tuple[int, int]:
        return self._torch.cuda.mem_get_info(index)

    def cpu(self) -> DeviceInfo:
        try:
            import psutil

            total = round(psutil.virtual_memory().total / GIB, 2)
        except ImportError:
            total = None
        return DeviceInfo(id="cpu", type="cpu", backend="cpu", name=platform.processor() or platform.machine(),
                          total_memory_gb=total, bf16=True, fp16=True)


class DeviceManager:
    def __init__(self, probe: DeviceProbe | None = None) -> None:
        self._probe = probe or TorchDeviceProbe()
        self._devices: list[DeviceInfo] | None = None

    def list_devices(self) -> list[DeviceInfo]:
        """Accelerators first (in index order), then CPU."""
        if self._devices is None:
            n = self._probe.accelerator_count()
            self._devices = [self._probe.accelerator(i) for i in range(n)] + [self._probe.cpu()]
        return list(self._devices)

    def accelerators(self) -> list[DeviceInfo]:
        return [d for d in self.list_devices() if d.is_accelerator]

    def resolve(self, spec: str | None = "auto") -> DeviceInfo:
        """`auto` → first accelerator, else CPU. `cuda` → `cuda:0`. Unknown or absent devices raise."""
        spec = (spec or "auto").strip().lower()
        devices = self.list_devices()
        available = ", ".join(d.id for d in devices)
        if spec == "auto":
            return devices[0]
        m = _DEVICE_RE.match(spec)
        if not m:
            raise DeviceNotSupportedError(
                f"Unsupported device {spec!r}. Available on this machine: {available}.",
                hint="Use auto, cpu, cuda or cuda:<index>. ROCm GPUs are addressed as cuda:<index>.")
        kind, index = m.group(1), m.group(2)
        if kind == "cpu":
            if index not in (None, "0"):
                raise DeviceNotSupportedError(f"Unsupported device {spec!r}: there is only one CPU device.")
            return devices[-1]
        wanted = f"cuda:{index or 0}"
        for d in devices:
            if d.id == wanted:
                return d
        raise DeviceNotSupportedError(
            f"Device {spec!r} is not present. Available on this machine: {available}.",
            hint="Check the NVIDIA/ROCm driver and that PyTorch was installed with GPU support "
                 "(python scripts/check_env.py).")

    def memory_bytes(self, device: DeviceInfo) -> tuple[int | None, int | None]:
        """(free, total) bytes of the device; system RAM for cpu."""
        if device.is_accelerator:
            return self._probe.mem_get_info(device.index or 0)
        try:
            import psutil

            vm = psutil.virtual_memory()
            return vm.available, vm.total
        except ImportError:
            return None, None

    def memory_info(self, device: DeviceInfo) -> dict[str, float | None]:
        """Free/total memory of the device in GiB (system RAM for cpu)."""
        free, total = self.memory_bytes(device)
        gib = lambda b: None if b is None else round(b / GIB, 2)  # noqa: E731
        return {"free_gb": gib(free), "total_gb": gib(total)}


_default: DeviceManager | None = None


def default_device_manager() -> DeviceManager:
    global _default
    if _default is None:
        _default = DeviceManager()
    return _default
