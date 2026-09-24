"""MOVA Runtime: how models execute (device, precision, memory, backend). See docs/architecture.md."""

from .base import Availability, ExecutionContext, RunResult, Runtime
from .device import DeviceInfo, DeviceManager, default_device_manager
from .manager import DEFAULT_RUNTIME, describe_runtimes, get_runtime, runtime_names
from .memory import MemoryManager
from .precision import Precision, PrecisionManager

__all__ = ["Availability", "ExecutionContext", "RunResult", "Runtime", "DeviceInfo", "DeviceManager",
           "default_device_manager", "DEFAULT_RUNTIME", "describe_runtimes", "get_runtime", "runtime_names",
           "MemoryManager", "Precision", "PrecisionManager"]
