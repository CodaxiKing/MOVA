"""Model registry: name/alias → model class + ModelSpec. Adding a backbone = one module that registers itself;
core, CLI and API do not change."""

from __future__ import annotations

from typing import Any

from common.errors import ModelNotFoundError

from .base import ModelSpec, MotionModel

DEFAULT_MODEL = "wan2.1-vace-1.3b"

_MODELS: dict[str, type[MotionModel]] = {}
_ALIASES: dict[str, str] = {}
_builtins_loaded = False


def register_model(cls: type[MotionModel]) -> type[MotionModel]:
    name = cls.spec.name
    if name in _MODELS and _MODELS[name] is not cls:
        raise ValueError(f"Model {name!r} is already registered by {_MODELS[name].__module__}")
    for key in (name, *cls.spec.aliases):
        owner = _ALIASES.get(key)
        if owner is not None and owner != name:
            raise ValueError(f"Alias {key!r} already points to {owner!r}")
    _MODELS[name] = cls
    for key in (name, *cls.spec.aliases):
        _ALIASES[key] = name
    return cls


def unregister_model(name: str) -> None:
    """For tests that register temporary models."""
    cls = _MODELS.pop(name, None)
    if cls is not None:
        for key in (name, *cls.spec.aliases):
            _ALIASES.pop(key, None)


def _load_builtins() -> None:
    global _builtins_loaded
    if not _builtins_loaded:
        _builtins_loaded = True
        from .backbones import wan_vace  # noqa: F401  (registers on import)


def model_names() -> list[str]:
    _load_builtins()
    return sorted(_MODELS)


def resolve_name(name: str | None) -> str:
    _load_builtins()
    key = (name or DEFAULT_MODEL).strip().lower()
    if key not in _ALIASES:
        raise ModelNotFoundError(f"Unknown model {name!r}. Registered models: {', '.join(model_names())}.",
                                 hint="List them with: mova info")
    return _ALIASES[key]


def get_model_class(name: str | None) -> type[MotionModel]:
    return _MODELS[resolve_name(name)]


def get_spec(name: str | None) -> ModelSpec:
    return get_model_class(name).spec


def list_specs() -> list[ModelSpec]:
    return [_MODELS[n].spec for n in model_names()]


def create_model(name: str | None, config: dict[str, Any]) -> MotionModel:
    return get_model_class(name)(config)
