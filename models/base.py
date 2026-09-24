"""Model interface: WHAT is computed. A model never picks a device or dtype; it receives an ExecutionContext
and a Runtime (runtime/base.py) and asks the runtime to place and run it.

Lifecycle used by core/inference.py:

    model = registry.create(name, config)
    model.configure(ctx, reference_size)      # resolve settings, no heavy work
    model.weights_status() / fetch_weights()  # verify / download (download only when authorised)
    model.resource_checks(ctx, weights, out)  # disk/RAM/VRAM preflight
    model.load(runtime, ctx)
    frames, stats = model.generate(reference, control)
    model.unload()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar

from PIL import Image


@dataclass(frozen=True)
class ModelSpec:
    """Registry metadata. Single source for what `mova info --model` prints and what validation checks."""

    name: str
    version: str
    family: str
    task: str
    description: str
    license: str
    default_config: str
    runtimes: tuple[str, ...]
    devices: tuple[str, ...]                      # device types: "cuda", "cpu"
    precisions: tuple[str, ...]
    capabilities: tuple[str, ...]
    requirements: tuple[str, ...] = ()            # importable Python modules
    weights: str | None = None                    # "repo@revision"
    weights_size_gb: float | None = None
    aliases: tuple[str, ...] = ()
    opt_in_devices: dict[str, str] = field(default_factory=dict)   # device type -> why it needs --allow-<type>
    verification: dict[str, str] = field(default_factory=dict)     # "runtime/device" -> evidence status

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WeightsStatus:
    complete: bool
    missing_bytes: int = 0
    revision: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def summary(self) -> str:
        return self.message or ("complete" if self.complete else f"missing {self.missing_bytes / 1e9:.2f} GB")


class MotionModel(ABC):
    """reference image + per-frame control → video frames."""

    spec: ClassVar[ModelSpec]

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.settings: Any = None

    # --- planning (cheap) -------------------------------------------------
    @abstractmethod
    def configure(self, ctx, reference_size: tuple[int, int]) -> Any:
        """Resolve settings for this context. The result must expose width, height, num_frames, fps."""

    @abstractmethod
    def weights_status(self, verify_hashes: bool = False) -> WeightsStatus: ...

    @abstractmethod
    def fetch_weights(self, verify_hashes: bool = False) -> WeightsStatus: ...

    def resource_checks(self, ctx, weights: WeightsStatus, output_dir) -> list:
        return []

    def record_fields(self) -> dict[str, Any]:
        """Extra fields for run.json (model id, revision, cache state...)."""
        return {"model": self.spec.name}

    def settings_dict(self) -> dict[str, Any]:
        return asdict(self.settings)

    # --- execution --------------------------------------------------------
    @abstractmethod
    def load(self, runtime, ctx) -> None: ...

    @abstractmethod
    def generate(self, reference: Image.Image, control: list[Image.Image]) -> tuple[list[Any], dict[str, Any]]: ...

    @abstractmethod
    def unload(self) -> None: ...

    @property
    @abstractmethod
    def loaded(self) -> bool: ...
