"""Automatic experiment registry: every run writes experiments/runs/<run_id>/run.json."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT
from .provenance import environment_fingerprint

RUNS_DIR = PROJECT_ROOT / "experiments" / "runs"

# Fields requested by the project spec; missing ones are recorded as null, never invented.
STANDARD_FIELDS = (
    "model", "dataset", "resolution", "frames", "batch", "learning_rate", "optimizer",
    "scheduler", "vram_peak_gb", "training_time_s", "loss", "checkpoint",
)


@dataclass
class ExperimentRun:
    """run.json always lives in runs_dir/<run_id>; heavy artifacts go to artifacts_root/<run_id> if given."""

    kind: str
    runs_dir: Path = RUNS_DIR
    artifacts_root: Path | None = None
    run_id: str = ""
    record: dict[str, Any] = field(default_factory=dict)
    _t0: float = 0.0

    def __post_init__(self) -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_id = self.run_id or f"{stamp}-{self.kind}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._t0 = time.perf_counter()
        self.record = {
            "run_id": self.run_id,
            "kind": self.kind,
            "artifacts_dir": str(self.artifacts_dir),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            **{k: None for k in STANDARD_FIELDS},
        }
        env = environment_fingerprint()
        self.record["git_commit"] = env["git"]["commit"]
        self.record["git_dirty"] = env["git"]["dirty"]
        self.record["environment"] = env
        self.save()

    @property
    def dir(self) -> Path:
        return Path(self.runs_dir) / self.run_id

    @property
    def artifacts_dir(self) -> Path:
        return Path(self.artifacts_root) / self.run_id if self.artifacts_root else self.dir

    def log(self, **kwargs: Any) -> None:
        self.record.update(kwargs)
        self.save()

    def finish(self, status: str = "success", **kwargs: Any) -> None:
        self.record.update(kwargs)
        self.record["status"] = status
        self.record["finished_at"] = datetime.now(timezone.utc).isoformat()
        self.record["wall_time_s"] = round(time.perf_counter() - self._t0, 2)
        self.save()

    def save(self) -> None:
        with (self.dir / "run.json").open("w", encoding="utf-8") as f:
            json.dump(self.record, f, indent=2, default=str)
