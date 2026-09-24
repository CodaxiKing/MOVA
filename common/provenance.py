"""Exact software provenance recorded with every run: packages, Python, git state."""

from __future__ import annotations

import importlib.metadata
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT

# Packages whose version can change generated or evaluated results.
TRACKED_PACKAGES = (
    "torch", "diffusers", "transformers", "accelerate", "huggingface_hub", "safetensors", "sentencepiece",
    "mediapipe", "numpy", "opencv-python", "pillow", "imageio", "imageio-ffmpeg", "pyyaml",
)
LOCK_FILE = PROJECT_ROOT / "requirements.lock.txt"


def package_versions(names: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def _git(*args: str) -> str | None:
    try:
        res = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return res.stdout.strip() if res.returncode == 0 else None


def git_state() -> dict[str, Any]:
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"commit": _git("rev-parse", "HEAD"), "dirty": None if status is None else bool(status)}


def _base_version(version: str) -> str:
    """'2.14.0+cu128' -> '2.14.0': the local build tag depends on CPU/GPU, not on the code version."""
    return version.split("+", 1)[0]


def read_lock(path: Path = LOCK_FILE) -> dict[str, str]:
    pins: dict[str, str] = {}
    if not path.exists():
        return pins
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        m = re.fullmatch(r"([A-Za-z0-9_.\-]+)==([^\s;]+)", line)
        if m:
            pins[m.group(1).lower().replace("_", "-")] = m.group(2)
    return pins


def lock_mismatches(installed: dict[str, str | None] | None = None, path: Path = LOCK_FILE) -> dict[str, dict]:
    """Tracked packages whose installed version differs from requirements.lock.txt."""
    pins = read_lock(path)
    installed = installed if installed is not None else package_versions()
    diff = {}
    for name, version in installed.items():
        pinned = pins.get(name.lower().replace("_", "-"))
        if pinned is None:
            continue
        if version is None or _base_version(version) != _base_version(pinned):
            diff[name] = {"locked": pinned, "installed": version}
    return diff


def environment_fingerprint() -> dict[str, Any]:
    packages = package_versions()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "lock_file": LOCK_FILE.name if LOCK_FILE.exists() else None,
        "lock_mismatches": lock_mismatches(packages),
        "git": git_state(),
    }
