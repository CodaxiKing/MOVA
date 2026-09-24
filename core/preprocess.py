"""Motion extraction service (MediaPipe, CPU) shared by `mova preprocess` and scripts/extract_motion.py."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

from common.config import load_config, resolve_path
from common.errors import ConfigError, InvalidInputError
from common.experiment import ExperimentRun


def run_extraction(video: str, out: str, config: str = "configs/extraction.yaml",
                   overrides: list[str] | None = None, retarget_to: str | None = None) -> tuple[dict[str, Any], Path]:
    from preprocessing.pipeline import ExtractionConfig, extract_motion

    try:
        raw = load_config(config, overrides)
    except FileNotFoundError as e:
        raise ConfigError(str(e)) from e
    known = {f.name for f in fields(ExtractionConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"Unknown extraction config keys: {sorted(unknown)}")
    cfg = ExtractionConfig(**raw)

    path = resolve_path(video)
    if not path.is_file():
        raise InvalidInputError(f"Video not found: {path}")
    run = ExperimentRun("extract")
    run.log(dataset=str(video), model="mediapipe", config=raw, output_dir=str(resolve_path(out)))
    try:
        summary = extract_motion(path, resolve_path(out), cfg)
    except Exception as e:
        run.finish("failed", error=repr(e))
        raise
    if retarget_to:
        import numpy as np
        from PIL import Image

        from preprocessing.retarget import retarget_directory

        ref = resolve_path(retarget_to)
        if not ref.is_file():
            run.finish("failed", error=f"reference not found: {ref}")
            raise InvalidInputError(f"Reference image not found: {ref}")
        with Image.open(ref) as im:
            rgb = np.asarray(im.convert("RGB"))
        try:
            summary["retarget"] = retarget_directory(resolve_path(out), rgb, resolve_path(out) / "retargeted")
        except ValueError as e:
            run.finish("failed", error=repr(e))
            raise InvalidInputError(f"Retargeting failed: {e}") from e
    run.finish("success", frames=summary["num_frames"], resolution=f"{summary['width']}x{summary['height']}",
               summary=summary)
    return summary, run.dir / "run.json"
