"""Numeric precision policy: which modes a device supports and what `auto` means.

Only fp32 / fp16 / bf16 exist. INT8 or other quantisation modes are not listed until they are implemented
and measured. Support is decided from the DeviceInfo (filled by DeviceManager), so models never ask the
hardware directly.
"""

from __future__ import annotations

from enum import Enum

from common.errors import PrecisionNotSupportedError

from .device import DeviceInfo

# Old configs used torch-style names; accept both.
_ALIASES = {"float32": "fp32", "fp32": "fp32", "float16": "fp16", "half": "fp16", "fp16": "fp16",
            "bfloat16": "bf16", "bf16": "bf16"}


class Precision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"

    @property
    def torch_name(self) -> str:
        return {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}[self.value]

    @classmethod
    def parse(cls, name: str) -> "Precision":
        key = _ALIASES.get(str(name).strip().lower())
        if key is None:
            raise PrecisionNotSupportedError(
                f"Unknown precision {name!r}.", hint="Use auto, fp32, fp16 or bf16.")
        return cls(key)


class PrecisionManager:
    def supported(self, device: DeviceInfo) -> list[Precision]:
        modes = [Precision.FP32]
        if device.fp16:
            modes.append(Precision.FP16)
        if device.bf16:
            modes.append(Precision.BF16)
        return modes

    def default(self, device: DeviceInfo) -> Precision:
        """GPU: bf16 when supported (Ampere+), else fp16. CPU: fp32 (diffusion on CPU is a test path)."""
        if not device.is_accelerator:
            return Precision.FP32
        return Precision.BF16 if device.bf16 else Precision.FP16

    def resolve(self, requested: str | Precision | None, device: DeviceInfo,
                allowed: tuple[str, ...] | None = None) -> Precision:
        """`allowed` restricts to what a model supports (from its registry spec)."""
        if requested in (None, "auto"):
            p = self.default(device)
            if allowed is not None and p.value not in allowed:
                p = next((Precision(a) for a in allowed if Precision(a) in self.supported(device)), p)
        else:
            p = requested if isinstance(requested, Precision) else Precision.parse(requested)
        if p not in self.supported(device):
            raise PrecisionNotSupportedError(
                f"Precision {p.value} is not supported on {device.id} ({device.name}). "
                f"Supported there: {', '.join(x.value for x in self.supported(device))}.",
                hint="Use --precision auto to pick the best supported mode.")
        if allowed is not None and p.value not in allowed:
            raise PrecisionNotSupportedError(
                f"Precision {p.value} is not supported by this model. Model supports: {', '.join(allowed)}.")
        return p
