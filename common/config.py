"""YAML config loading with dotted-key CLI overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for item in overrides or []:
        apply_override(cfg, item)
    return cfg


def apply_override(cfg: dict[str, Any], item: str) -> None:
    """Apply `a.b.c=value`; value is parsed as YAML so numbers/bools/null work."""
    if "=" not in item:
        raise ValueError(f"Override must be key=value, got: {item!r}")
    key, raw = item.split("=", 1)
    node = cfg
    parts = key.strip().split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"Cannot override {key!r}: {part!r} is not a mapping")
    node[parts[-1]] = yaml.safe_load(raw)


def resolve_path(p: str | Path) -> Path:
    """Resolve project-relative paths; absolute paths are kept as-is."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p
