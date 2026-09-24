"""Runtime interface: HOW a model executes (device, precision, placement, memory), not WHAT it computes.

Core and models talk to this interface; only concrete backends import a framework. A model receives an
ExecutionContext already validated for device/precision/offload, asks the runtime to place its modules and
runs its forward pass through `run()`, which records time and memory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, TypeVar

from .device import DeviceInfo
from .memory import MemoryReport
from .precision import Precision

T = TypeVar("T")


@dataclass(frozen=True)
class Availability:
    available: bool
    reason: str = ""
    version: str | None = None


@dataclass(frozen=True)
class ExecutionContext:
    runtime: str
    device: DeviceInfo
    precision: Precision
    offload: str  # "none" | "model" | "sequential"

    def to_dict(self) -> dict[str, Any]:
        return {"runtime": self.runtime, "device": self.device.id, "device_name": self.device.name,
                "backend": self.device.backend, "precision": self.precision.value, "offload": self.offload}


@dataclass
class RunResult:
    value: Any
    memory: MemoryReport
    extra: dict[str, Any] = field(default_factory=dict)


class Runtime(ABC):
    name: ClassVar[str]

    @classmethod
    @abstractmethod
    def availability(cls) -> Availability:
        """Whether the backend can run here (package importable, driver present, ...)."""

    @abstractmethod
    def devices(self) -> list[DeviceInfo]: ...

    @abstractmethod
    def context(self, device: str | None = "auto", precision: str | None = "auto", offload: str | None = "auto",
                *, allowed_precisions: tuple[str, ...] | None = None) -> ExecutionContext:
        """Resolve `auto` values and validate the combination; raise a structured error otherwise."""

    @abstractmethod
    def dtype(self, ctx: ExecutionContext) -> Any:
        """Framework dtype object for ctx.precision."""

    @abstractmethod
    def place(self, obj: Any, ctx: ExecutionContext) -> Any:
        """Move/offload a loaded model object according to ctx. Returns the placed object."""

    @abstractmethod
    def run(self, fn: Callable[[], T], ctx: ExecutionContext) -> RunResult:
        """Execute an inference callable without autograd, tracking wall time and memory."""

    @abstractmethod
    def generator(self, seed: int, ctx: ExecutionContext) -> Any:
        """Seeded RNG for sampling (reproducible across devices)."""

    @abstractmethod
    def release(self, ctx: ExecutionContext) -> None:
        """Free cached memory after a model was dropped."""

    def memory_stats(self, ctx: ExecutionContext) -> dict[str, Any]:
        return {}

    def info(self) -> dict[str, Any]:
        a = self.availability()
        return {"name": self.name, "available": a.available, "reason": a.reason, "version": a.version}
