"""Structured errors shared by runtime, models, core, CLI and API.

Every error explains what was requested, what is available and what to do, so incompatibilities surface
before execution instead of as an obscure traceback. `code` is stable for API responses and exit codes.
"""

from __future__ import annotations


class MovaError(Exception):
    code = "mova_error"
    exit_code = 2

    def __init__(self, message: str, *, hint: str | None = None):
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        return self.message if not self.hint else f"{self.message}\nHint: {self.hint}"

    def to_dict(self) -> dict:
        return {"error": self.code, "message": self.message, "hint": self.hint}


class InvalidInputError(MovaError, ValueError):
    code = "invalid_input"


class ConfigError(MovaError, ValueError):
    code = "invalid_config"


class RuntimeNotAvailableError(MovaError):
    code = "runtime_not_available"


class ModelNotFoundError(MovaError, KeyError):
    code = "model_not_found"

    def __str__(self) -> str:  # KeyError would otherwise repr() the message
        return MovaError.__str__(self)


class ModelRuntimeIncompatibleError(MovaError):
    code = "model_runtime_incompatible"


class DeviceNotSupportedError(MovaError):
    code = "device_not_supported"


class PrecisionNotSupportedError(MovaError):
    code = "precision_not_supported"


class InsufficientResourcesError(MovaError, RuntimeError):
    code = "insufficient_resources"


class InsufficientVRAMError(InsufficientResourcesError):
    code = "insufficient_vram"


class ModelWeightsMissingError(MovaError):
    code = "model_weights_missing"
