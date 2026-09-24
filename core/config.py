"""Run configuration: pick the model and its config file, and resolve the runtime section with one precedence.

Precedence for runtime/device/precision/offload (highest first):
  explicit request values (CLI flags / API fields)
  > `runtime:` section of the run config (including --set runtime.<key>=...)
  > legacy `model.dtype` / `model.offload` (configs written before ADR-009, e.g. old benchmark runs)
  > configs/runtime.yaml
"""

from __future__ import annotations

from typing import Any

from common.config import load_config
from common.errors import ConfigError
from common.logging_utils import get_logger
from models.registry import DEFAULT_MODEL, get_spec, resolve_name

log = get_logger("mova.core.config")

RUNTIME_KEYS = ("runtime", "device", "precision", "offload")
RUNTIME_CONFIG = "configs/runtime.yaml"
_LEGACY = {"dtype": "precision", "offload": "offload"}


def _load(path: str, overrides: list[str] | None = None) -> dict[str, Any]:
    try:
        return load_config(path, overrides)
    except FileNotFoundError as e:
        raise ConfigError(str(e)) from e
    except ValueError as e:
        raise ConfigError(f"Invalid override: {e}") from e


def load_runtime_defaults(path: str = RUNTIME_CONFIG) -> dict[str, Any]:
    raw = _load(path)
    unknown = set(raw) - set(RUNTIME_KEYS)
    if unknown:
        raise ConfigError(f"Unknown keys in {path}: {sorted(unknown)}. Allowed: {list(RUNTIME_KEYS)}.")
    return {k: raw.get(k, "auto") for k in RUNTIME_KEYS}


def resolve_run_config(*, model: str | None = None, config: str | None = None, overrides: list[str] | None = None,
                       runtime_overrides: dict[str, str | None] | None = None,
                       runtime_config: str = RUNTIME_CONFIG) -> tuple[str, dict[str, Any]]:
    """Return (registered model name, merged config with a complete `runtime` section)."""
    explicit_config = config is not None
    name = resolve_name(model) if model else None
    if config is None:
        config = get_spec(name or DEFAULT_MODEL).default_config
    cfg = _load(config, overrides)
    cfg_name = (cfg.get("model") or {}).get("name")
    if name and explicit_config and cfg_name and resolve_name(cfg_name) != name:
        raise ConfigError(f"--model {model!r} conflicts with model.name {cfg_name!r} in {config}.",
                          hint="Drop --config to use the model's default config, or drop --model.")
    name = name or resolve_name(cfg_name or DEFAULT_MODEL)
    cfg.setdefault("model", {})["name"] = name

    merged = load_runtime_defaults(runtime_config)
    for old, new in _LEGACY.items():
        value = cfg["model"].pop(old, None)
        if value not in (None, "auto"):
            log.warning("model.%s is deprecated (ADR-009); use runtime.%s. Using %r.", old, new, value)
            merged[new] = value
    section = cfg.get("runtime") or {}
    if not isinstance(section, dict):
        raise ConfigError("`runtime` in the run config must be a mapping")
    unknown = set(section) - set(RUNTIME_KEYS)
    if unknown:
        raise ConfigError(f"Unknown runtime keys {sorted(unknown)}. Allowed: {list(RUNTIME_KEYS)}.")
    merged.update({k: v for k, v in section.items() if v is not None})
    merged.update({k: v for k, v in (runtime_overrides or {}).items() if v is not None})
    cfg["runtime"] = merged
    return name, cfg
