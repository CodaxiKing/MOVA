"""Training service behind `mova train` (ADR-009: interfaces call core, core calls runtime/models/training).

`smoke=True` trains the tiny random backbone on a synthetic manifest in a temp folder: it proves the loop on any
machine and is what CI/CPU can run. A real run reads configs/train_adapter.yaml, refuses identities that belong
to the benchmark, and records everything in experiments/runs/<id>/run.json (standard fields: learning_rate,
optimizer, loss, checkpoint, vram_peak_gb, training_time_s).
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from common.config import load_config, resolve_path
from common.errors import ConfigError
from common.experiment import RUNS_DIR, ExperimentRun


@dataclass
class TrainRequest:
    config: str | None = "configs/train_adapter.yaml"
    smoke: bool = False
    steps: int | None = None
    resume: str | None = None
    runtime: str | None = None
    device: str | None = None
    precision: str | None = None
    overrides: list[str] | None = None
    runs_dir: Path | None = None


def benchmark_identities(manifest: str | Path) -> set[str]:
    path = resolve_path(manifest)
    if not path.is_file():
        raise ConfigError(f"forbid_identities_from: {path} not found")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {c["identity_id"] for c in data.get("cases", []) if c.get("identity_id")}


SMOKE = {"backbone": "tiny", "steps": 20, "batch_size": 2, "save_every": 0, "reference_size": 32, "max_references": 2,
         "conditioning": {"cond_dim": 32, "adapter_dim": 16, "heads": 4, "identity_patch": 8},
         "train": {"lr": 3e-3, "grad_accum": 1}}


def run_training(req: TrainRequest, echo: Callable[[str], None] = print) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    from runtime import get_runtime
    from training.backbone import load_backbone
    from training.dataset import MotionClipDataset, collate
    from training.trainer import AdapterTrainer, TrainConfig, conditioning_config_from, write_history

    tmp = None
    if req.smoke:
        from training.smoke import make_synthetic_dataset

        tmp = tempfile.TemporaryDirectory(prefix="mova-smoke-")
        cfg = {**SMOKE, "dataset": str(make_synthetic_dataset(tmp.name, samples=2)), "split": "train",
               "output_dir": str(Path(tmp.name) / "ckpt")}
    else:
        try:
            cfg = load_config(req.config, req.overrides)
        except FileNotFoundError as e:
            raise ConfigError(str(e)) from e
    steps = int(req.steps or cfg["steps"])
    rt = get_runtime(req.runtime)
    ctx = rt.context(req.device or "auto", req.precision or ("fp32" if req.smoke else "auto"), "none")
    forbid = benchmark_identities(cfg["forbid_identities_from"]) if cfg.get("forbid_identities_from") else set()
    manifest = resolve_path(cfg["dataset"])
    if not manifest.is_file():
        raise ConfigError(f"Training manifest not found: {manifest}",
                          hint="Build it with scripts/datasets.py and training/precompute.py (docs/training.md).")
    ds = MotionClipDataset(manifest, cfg.get("split", "train"), max_references=int(cfg["max_references"]),
                           reference_size=int(cfg["reference_size"]), forbid_identities=forbid)
    transformer, extra = load_backbone(cfg["backbone"], torch.float32)
    from models.adapters.stack import MovaConditioning

    cond = MovaConditioning.for_transformer(transformer, conditioning_config_from(cfg.get("conditioning")))
    tcfg = TrainConfig(**(cfg.get("train") or {}))
    trainer = AdapterTrainer(transformer, cond, tcfg, runtime=rt, ctx=ctx, forward_extra=extra)
    ckpt = resolve_path(cfg["output_dir"]) / "adapter.pt"
    if req.resume:
        trainer.load(resolve_path(req.resume))
        echo(f"Resumed from {req.resume} at step {trainer.step_count}")
    run = ExperimentRun("train", runs_dir=req.runs_dir or RUNS_DIR)
    n_train = sum(p.numel() for p in trainer.params)
    run.log(model=f"MOVA adapter on {cfg['backbone']}", dataset=str(manifest), batch=int(cfg["batch_size"]),
            learning_rate=tcfg.lr, optimizer="AdamW", scheduler=None, runtime=ctx.to_dict(),
            train_config=tcfg.to_dict(), conditioning_config=cond.cfg.to_dict(), trainable_parameters=n_train,
            samples=len(ds), smoke=req.smoke)
    echo(f"Training {n_train:,} parameters on {ctx.device.id} ({ctx.precision.value}); backbone {cfg['backbone']} frozen; "
         f"{len(ds)} samples; {steps} steps")
    loader = DataLoader(ds, batch_size=int(cfg["batch_size"]), shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(tcfg.seed))
    try:
        with rt.memory_manager.track(ctx.device) as mem:
            history = trainer.fit(loader, steps, log=lambda r: echo(f"step {r['step']:>5}  loss {r['loss']:.4f}"),
                                  checkpoint=ckpt, save_every=int(cfg.get("save_every") or 0))
    except BaseException as e:
        run.finish("failed", error=repr(e), checkpoint=str(ckpt) if ckpt.exists() else None)
        raise
    write_history(run.dir / "history.jsonl", history)
    result = {"steps": trainer.step_count, "first_loss": history[0]["loss"] if history else None,
              "last_loss": history[-1]["loss"] if history else None, "checkpoint": str(ckpt),
              "record": str(run.dir / "run.json"), "vram_peak_gb": mem.peak_allocated_gb}
    run.finish("success", loss=result["last_loss"], checkpoint=None if req.smoke else str(ckpt),
               vram_peak_gb=mem.peak_allocated_gb, training_time_s=mem.wall_time_s, history=str(run.dir / "history.jsonl"),
               first_loss=result["first_loss"])
    if tmp is not None:
        result["checkpoint"] = "(smoke: discarded with the temporary folder)"
        tmp.cleanup()
    return result
