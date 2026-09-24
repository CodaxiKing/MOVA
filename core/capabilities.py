"""Capability validation: is (model, runtime, device, precision, offload) runnable HERE, before any heavy work?

Order of checks: runtime exists and is available → model supports the runtime → model's Python requirements
are importable → device exists → precision supported by device and model → offload valid for the device →
model supports the device type → opt-in devices (e.g. CPU for the 1.3B model) were explicitly allowed.
Memory sufficiency is checked later by the model's resource preflight (needs resolved settings).
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

from common.errors import DeviceNotSupportedError, ModelRuntimeIncompatibleError, MovaError, RuntimeNotAvailableError
from models.base import ModelSpec
from runtime import ExecutionContext, Runtime, get_runtime
from runtime.manager import NOT_IMPLEMENTED, runtime_names


def missing_requirements(spec: ModelSpec) -> list[str]:
    return [m for m in spec.requirements if importlib.util.find_spec(m) is None]


def validate(spec: ModelSpec, runtime: str | None = None, device: str | None = "auto", precision: str | None = "auto",
             offload: str | None = "auto", *, allow_devices: tuple[str, ...] = (),
             runtime_obj: Runtime | None = None) -> tuple[Runtime, ExecutionContext]:
    """Return (runtime, validated context) or raise the first structured incompatibility."""
    rt = runtime_obj or get_runtime(runtime)
    if rt.name not in spec.runtimes:
        raise ModelRuntimeIncompatibleError(
            f"Model {spec.name!r} does not support runtime {rt.name!r}. Supported: {', '.join(spec.runtimes)}.")
    missing = missing_requirements(spec)
    if missing:
        raise RuntimeNotAvailableError(f"Model {spec.name!r} needs Python packages that are not installed: "
                                       f"{', '.join(missing)}.", hint="pip install -r requirements.txt")
    ctx = rt.context(device, precision, offload, allowed_precisions=spec.precisions)
    dtype = ctx.device.type
    if dtype not in spec.devices:
        raise DeviceNotSupportedError(f"Model {spec.name!r} does not run on {dtype} devices. "
                                      f"Supported device types: {', '.join(spec.devices)}.")
    if dtype in spec.opt_in_devices and dtype not in allow_devices:
        raise DeviceNotSupportedError(f"Device {ctx.device.id} for model {spec.name!r}: {spec.opt_in_devices[dtype]}.",
                                      hint=f"Run on a GPU, or pass --allow-{dtype} to force it.")
    return rt, ctx


@dataclass
class CompatibilityRow:
    runtime: str
    device: str
    status: str            # "compatible" | "opt-in" | "incompatible" | "unavailable"
    precision: str | None
    offload: str | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def compatibility_matrix(spec: ModelSpec) -> list[CompatibilityRow]:
    """Every known runtime x every device of this machine, with the auto-selected precision/offload."""
    rows: list[CompatibilityRow] = []
    for name in runtime_names() + sorted(NOT_IMPLEMENTED):
        try:
            rt = get_runtime(name)
        except MovaError as e:
            rows.append(CompatibilityRow(name, "*", "unavailable", None, None, e.message))
            continue
        for dev in rt.devices():
            try:
                _, ctx = validate(spec, device=dev.id, runtime_obj=rt, allow_devices=tuple(spec.devices))
                opt = spec.opt_in_devices.get(dev.type)
                verified = spec.verification.get(f"{name}/{dev.type}", "UNVERIFIED")
                rows.append(CompatibilityRow(name, dev.id, "opt-in" if opt else "compatible", ctx.precision.value,
                                             ctx.offload, f"{opt}. {verified}" if opt else verified))
            except MovaError as e:
                rows.append(CompatibilityRow(name, dev.id, "incompatible", None, None, e.message))
    return rows
