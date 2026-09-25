"""Adapter training loop: frozen Wan DiT + trainable MOVA conditioning (encoders, fusion, adapter).

Only the conditioning parameters are optimised; the backbone is frozen (requires_grad False, eval) and never
saved. Device and precision come from the runtime ExecutionContext (ADR-009): fp32 runs plainly, fp16/bf16 run
under torch.autocast. Checkpoints hold the trainable state, the optimizer, the step and both configs, so a run
resumes exactly (tested). Validated on CPU with a tiny random DiT only; real training needs the GPU (Phase 5).
"""

from __future__ import annotations

import json
import time
from contextlib import ExitStack, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from models.adapters.stack import ConditioningConfig, MovaConditioning

from .losses import (NUM_TRAIN_TIMESTEPS, flow_matching_loss, flow_matching_pair, keypoint_region_mask,
                     region_weights, sample_sigmas)


@dataclass
class TrainConfig:
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_accum: int = 1
    max_grad_norm: float = 1.0
    flow_shift: float = 3.0
    face_weight: float = 2.0
    hand_weight: float = 2.0
    seed: int = 0
    gradient_checkpointing: bool = False  # recompute backbone activations in backward (needed on 8 GB GPUs)

    def to_dict(self) -> dict:
        return asdict(self)


def _to(batch: dict[str, torch.Tensor], device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


class AdapterTrainer:
    def __init__(self, transformer: torch.nn.Module, conditioning: MovaConditioning, cfg: TrainConfig | None = None,
                 *, runtime=None, ctx=None, forward_extra=None):
        """forward_extra(x_t) -> extra kwargs for the backbone (e.g. the null VACE control, training/backbone.py)."""
        self.cfg = cfg or TrainConfig()
        self.forward_extra = forward_extra
        self.runtime, self.ctx = runtime, ctx
        self.device = torch.device(ctx.device.id) if ctx is not None else torch.device("cpu")
        self.transformer = transformer.to(self.device).eval()
        for p in self.transformer.parameters():
            p.requires_grad_(False)
        self.cond = conditioning.to(self.device).train()
        self.cond.adapter.attach(self.transformer)
        if self.cfg.gradient_checkpointing:
            if not hasattr(self.transformer, "enable_gradient_checkpointing"):
                raise ValueError("gradient_checkpointing needs a Diffusers backbone")
            self.transformer.enable_gradient_checkpointing()
        self.params = [p for p in self.cond.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(self.params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        # fp16 has a narrow exponent: without loss scaling the small gradients of a zero-initialised adapter
        # underflow to 0. bf16/fp32 do not need it.
        fp16 = ctx is not None and ctx.precision.value == "fp16"
        self.scaler = torch.amp.GradScaler(self.device.type) if fp16 else None
        self.step_count = 0
        self._micro = 0
        self.generator = torch.Generator().manual_seed(self.cfg.seed)

    def _autocast(self):
        if self.ctx is None or self.ctx.precision.value == "fp32":
            return nullcontext()
        return torch.autocast(self.device.type, dtype=self.runtime.dtype(self.ctx))

    def loss(self, batch: dict[str, torch.Tensor], *, sigma: torch.Tensor | None = None,
             noise: torch.Tensor | None = None, scope: ExitStack | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
        """scope: keep the conditioning active in the caller's ExitStack (train_step does, so that gradient
        checkpointing recomputes blocks with the adapter hooks live during backward)."""
        batch = _to(batch, self.device)
        x0 = batch["latents"]
        b, _, f, hl, wl = x0.shape
        num_frames, height, width = 4 * (f - 1) + 1, hl * 8, wl * 8
        sigma = sample_sigmas(b, self.cfg.flow_shift, self.generator) if sigma is None else sigma
        noise = torch.randn(x0.shape, generator=self.generator) if noise is None else noise
        sigma, noise = sigma.to(self.device), noise.to(self.device)
        xt, target = flow_matching_pair(x0, noise, sigma)
        regions = []
        for key, w in (("face_xy", self.cfg.face_weight), ("hands_xy", self.cfg.hand_weight)):
            if key in batch and w != 1.0:
                m = keypoint_region_mask(batch[key].cpu(), batch[f"{key}_mask"].cpu(), (hl, wl), f)
                regions.append((m, w))
        weights = region_weights(tuple(x0.shape), regions).to(self.device) if regions else None
        with ExitStack() as local:
            local.enter_context(self._autocast())
            (scope or local).enter_context(self.cond.conditioned(batch, num_frames, height, width))
            extra = self.forward_extra(xt) if self.forward_extra else {}
            pred = self.transformer(hidden_states=xt, timestep=sigma * NUM_TRAIN_TIMESTEPS,
                                    encoder_hidden_states=batch["prompt_embeds"], return_dict=False, **extra)[0]
        loss = flow_matching_loss(pred, target, weights)
        return loss, {"sigma_mean": float(sigma.mean())}

    def train_step(self, batch: dict[str, torch.Tensor], **kw) -> dict[str, float]:
        with ExitStack() as scope:
            loss, info = self.loss(batch, scope=scope, **kw)
            scaled = loss / self.cfg.grad_accum
            (self.scaler.scale(scaled) if self.scaler else scaled).backward()
        self._micro += 1
        out = {"loss": float(loss.detach()), **info, "updated": False}
        if self._micro % self.cfg.grad_accum == 0:
            if self.scaler:
                self.scaler.unscale_(self.opt)
            out["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(self.params, self.cfg.max_grad_norm))
            if self.scaler:
                self.scaler.step(self.opt)  # skipped (and the scale lowered) if grads are inf/nan
                self.scaler.update()
                out["loss_scale"] = float(self.scaler.get_scale())
            else:
                self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            self.step_count += 1
            out["updated"] = True
        return out

    def fit(self, batches: Iterable[dict[str, torch.Tensor]], steps: int, *, log=None,
            checkpoint: Path | None = None, save_every: int = 0) -> list[dict[str, float]]:
        history, t0 = [], time.perf_counter()
        it = iter(batches)
        while self.step_count < steps:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(batches)
                batch = next(it)
            rec = self.train_step(batch)
            if rec["updated"]:
                rec["step"] = self.step_count
                rec["elapsed_s"] = round(time.perf_counter() - t0, 3)
                history.append(rec)
                if log:
                    log(rec)
                if checkpoint and save_every and self.step_count % save_every == 0:
                    self.save(checkpoint)
        if checkpoint:
            self.save(checkpoint)
        return history

    def trainable_state(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.cond.state_dict().items()}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save({"format": "mova-adapter-v1", "step": self.step_count, "train_config": self.cfg.to_dict(),
                    "conditioning_config": self.cond.cfg.to_dict(), "conditioning": self.trainable_state(),
                    "optimizer": self.opt.state_dict(), "generator": self.generator.get_state(),
                    **({"scaler": self.scaler.state_dict()} if self.scaler else {})}, tmp)
        tmp.replace(path)
        return path

    def load(self, path: str | Path) -> None:
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if ck.get("format") != "mova-adapter-v1":
            raise ValueError(f"{path} is not a MOVA adapter checkpoint")
        if ck["conditioning_config"] != self.cond.cfg.to_dict():
            raise ValueError("Checkpoint was trained with a different conditioning config")
        self.cond.load_state_dict(ck["conditioning"])
        self.opt.load_state_dict(ck["optimizer"])
        self.step_count = ck["step"]
        self.generator.set_state(ck["generator"])
        if self.scaler and "scaler" in ck:
            self.scaler.load_state_dict(ck["scaler"])


def conditioning_config_from(d: dict | None) -> ConditioningConfig:
    d = dict(d or {})
    if "streams" in d:
        d["streams"] = tuple(d["streams"])
    return ConditioningConfig(**d)


def write_history(path: Path, history: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(h) for h in history) + "\n", encoding="utf-8")
