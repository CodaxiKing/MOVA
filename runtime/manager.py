"""Runtime lookup by name. Only backends that actually work are registered.

ONNX Runtime and TensorRT are known deployment targets but NOT implemented: the Wan VACE pipeline has no
validated export path. Asking for them raises RuntimeNotAvailableError with that explanation instead of
exposing placeholder classes (see docs/architecture.md → Deployment).
"""

from __future__ import annotations

from typing import Any

from common.errors import RuntimeNotAvailableError

from .backends.pytorch import PyTorchRuntime
from .base import Runtime

DEFAULT_RUNTIME = "pytorch"

_RUNTIMES: dict[str, type[Runtime]] = {PyTorchRuntime.name: PyTorchRuntime}

NOT_IMPLEMENTED = {
    "onnx": "ONNX export of the Wan VACE pipeline is not implemented or validated (Phase 13).",
    "tensorrt": "TensorRT requires a validated ONNX export and an NVIDIA GPU; not implemented (Phase 14).",
}


def register_runtime(cls: type[Runtime]) -> type[Runtime]:
    _RUNTIMES[cls.name] = cls
    return cls


def runtime_names() -> list[str]:
    return sorted(_RUNTIMES)


def get_runtime(name: str | None = None, **kwargs: Any) -> Runtime:
    name = (name or DEFAULT_RUNTIME).strip().lower()
    if name in NOT_IMPLEMENTED:
        raise RuntimeNotAvailableError(f"Runtime {name!r} is not available: {NOT_IMPLEMENTED[name]}",
                                       hint=f"Use --runtime {DEFAULT_RUNTIME}.")
    cls = _RUNTIMES.get(name)
    if cls is None:
        raise RuntimeNotAvailableError(f"Unknown runtime {name!r}. Known runtimes: {', '.join(runtime_names())}.")
    a = cls.availability()
    if not a.available:
        raise RuntimeNotAvailableError(f"Runtime {name!r} cannot run on this machine: {a.reason}")
    return cls(**kwargs)


def describe_runtimes() -> list[dict[str, Any]]:
    rows = []
    for name in runtime_names():
        a = _RUNTIMES[name].availability()
        rows.append({"name": name, "status": "available" if a.available else "unavailable",
                     "version": a.version, "reason": a.reason})
    rows += [{"name": n, "status": "not implemented", "version": None, "reason": r} for n, r in NOT_IMPLEMENTED.items()]
    return rows
