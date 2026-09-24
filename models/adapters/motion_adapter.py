"""Motion Adapter: zero-initialised residual injection into a FROZEN Wan DiT, one small module per block.

Two paths, both residual and zero-init (at step 0 the backbone output is bit-identical, tested):
- sparse (cross-attention): hidden tokens of the DiT attend to condition tokens (body/face/hands motion tokens
  of the SAME latent frame + global identity tokens). This is how face blendshapes and 3D hands reach the model
  instead of being thrown away after rendering the skeleton.
- dense (optional): a pose video encoded to the DiT token grid, added through a per-block zero-init linear
  (the VACE/ControlNet-style spatial path).

The adapter attaches with forward hooks on `transformer.blocks[i]` (diffusers WanTransformer3DModel and
WanVACETransformer3DModel both call `block(hidden, encoder_hidden, temb, rotary)`), so the backbone code and
weights are untouched and the adapter can be detached.

Token grid: latents (B, 16, F, H/8, W/8) with F = (num_frames - 1) / 4 + 1; patch (1, 2, 2) -> L = F * H/16 * W/16
tokens, frame-major. `latent_grid()` computes it; the hook checks it matches the hidden sequence length.
VACE prepends each reference image as an extra latent frame (found by the end-to-end test): pass
`reference_frames`; those frames attend only to global (identity) tokens and get no dense residual.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn
from torch.nn import functional as F

from models.fusion.condition import GLOBAL, ConditionTokens


def latent_grid(num_frames: int, height: int, width: int, patch: tuple[int, int, int] = (1, 2, 2),
                vae_spatial: int = 8, vae_temporal: int = 4, reference_frames: int = 0) -> tuple[int, int, int]:
    """(latent frames incl. prepended reference frames, token rows, token cols)."""
    if (num_frames - 1) % vae_temporal or height % (vae_spatial * patch[1]) or width % (vae_spatial * patch[2]):
        raise ValueError("num_frames must be 4k+1 and height/width multiples of 16")
    f = ((num_frames - 1) // vae_temporal + 1) // patch[0]
    return f + reference_frames, height // vae_spatial // patch[1], width // vae_spatial // patch[2]


def frame_mask(grid: tuple[int, int, int], cond: ConditionTokens, reference_frames: int = 0) -> torch.Tensor:
    """(B, L, M) bool: hidden token (video frame f) may attend to condition token j iff j is in frame f or global.
    Prepended reference frames get negative frame numbers and therefore see only global tokens."""
    f, hp, wp = grid
    q_frame = (torch.arange(f, device=cond.tokens.device) - reference_frames).repeat_interleave(hp * wp)
    allowed = (cond.frame_index[None, :] == q_frame[:, None]) | (cond.frame_index[None, :] == GLOBAL)
    return allowed[None] & cond.mask[:, None, :]


class BlockAdapter(nn.Module):
    """Residual cross-attention hidden -> condition tokens. out_proj is zero-initialised."""

    def __init__(self, hidden_dim: int, cond_dim: int, dim: int = 128, heads: int = 4):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.q = nn.Linear(hidden_dim, dim)
        self.kv = nn.Linear(cond_dim, 2 * dim)
        self.out = nn.Linear(dim, hidden_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, hidden: torch.Tensor, cond: ConditionTokens, allowed: torch.Tensor) -> torch.Tensor:
        b, length, _ = hidden.shape
        q = self.q(self.norm(hidden)).view(b, length, self.heads, -1).transpose(1, 2)
        k, v = self.kv(cond.tokens.to(hidden.dtype)).chunk(2, -1)
        k = k.view(b, -1, self.heads, q.shape[-1]).transpose(1, 2)
        v = v.view(b, -1, self.heads, q.shape[-1]).transpose(1, 2)
        none = ~allowed.any(-1)                       # queries with nothing to attend to (e.g. missing streams)
        mask = allowed | none[..., None]              # avoid all -inf rows; their output is zeroed below
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None])
        out = out.transpose(1, 2).reshape(b, length, -1)
        out = out.masked_fill(none[..., None], 0.0)
        return self.out(out)


class PoseGridEncoder(nn.Module):
    """Dense path: pose video (B, 3, T=4k+1, H, W) in [0, 1] -> (B, L, dim) on the DiT token grid."""

    def __init__(self, dim: int = 128, channels: int = 32):
        super().__init__()
        self.first = nn.Conv3d(3, channels, (1, 3, 3), padding=(0, 1, 1))
        self.down = nn.Sequential(  # spatial /16 (VAE 8 x patch 2)
            nn.Conv3d(channels, channels, (1, 4, 4), stride=(1, 4, 4)), nn.SiLU(),
            nn.Conv3d(channels, channels, (1, 4, 4), stride=(1, 4, 4)), nn.SiLU())
        self.temporal = nn.Conv3d(channels, dim, (4, 1, 1), stride=(4, 1, 1))   # 4 frames -> 1 latent frame
        self.head = nn.Conv3d(channels, dim, 1)                                   # frame 0 alone

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        x = self.down(F.silu(self.first(video * 2 - 1)))
        x = torch.cat([self.head(x[:, :, :1]), self.temporal(x[:, :, 1:])], 2)   # (B, dim, F, H/16, W/16)
        return x.flatten(2).transpose(1, 2)


class MotionAdapter(nn.Module):
    def __init__(self, hidden_dim: int, num_blocks: int, cond_dim: int, *, blocks: list[int] | None = None,
                 dim: int = 128, heads: int = 4, dense_dim: int | None = None):
        super().__init__()
        self.block_ids = list(range(num_blocks)) if blocks is None else list(blocks)
        if any(i < 0 or i >= num_blocks for i in self.block_ids):
            raise ValueError("adapter block index out of range")
        self.cross = nn.ModuleDict({str(i): BlockAdapter(hidden_dim, cond_dim, dim, heads) for i in self.block_ids})
        self.dense = None
        if dense_dim is not None:
            self.dense = nn.ModuleDict({str(i): nn.Linear(dense_dim, hidden_dim) for i in self.block_ids})
            for lin in self.dense.values():
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
        self.scale = nn.Parameter(torch.ones(len(self.block_ids)))
        self._handles: list = []
        self._cond: ConditionTokens | None = None
        self._dense_tokens: torch.Tensor | None = None
        self._grid: tuple[int, int, int] | None = None
        self._allowed: torch.Tensor | None = None

    # --- conditioning state -------------------------------------------------------------------------
    def set_condition(self, cond: ConditionTokens | None, grid: tuple[int, int, int],
                      dense_tokens: torch.Tensor | None = None, reference_frames: int = 0) -> None:
        if dense_tokens is not None and self.dense is None:
            raise ValueError("dense tokens given but the adapter has no dense path")
        if dense_tokens is not None and reference_frames:  # no pose for reference frames: zero residual there
            pad = dense_tokens.new_zeros(dense_tokens.shape[0], reference_frames * grid[1] * grid[2],
                                         dense_tokens.shape[2])
            dense_tokens = torch.cat([pad, dense_tokens], 1)
        self._cond, self._grid, self._dense_tokens = cond, grid, dense_tokens
        self._allowed = frame_mask(grid, cond, reference_frames) if cond is not None else None

    def clear_condition(self) -> None:
        self._cond = self._dense_tokens = self._grid = self._allowed = None

    @contextmanager
    def conditioned(self, cond: ConditionTokens | None, grid: tuple[int, int, int],
                    dense_tokens: torch.Tensor | None = None, reference_frames: int = 0) -> Iterator["MotionAdapter"]:
        self.set_condition(cond, grid, dense_tokens, reference_frames)
        try:
            yield self
        finally:
            self.clear_condition()

    # --- hooks ----------------------------------------------------------------------------------------
    def attach(self, transformer: nn.Module) -> "MotionAdapter":
        if self._handles:
            raise RuntimeError("adapter already attached")
        blocks = transformer.blocks
        if max(self.block_ids) >= len(blocks):
            raise ValueError(f"transformer has {len(blocks)} blocks")
        for pos, i in enumerate(self.block_ids):
            self._handles.append(blocks[i].register_forward_hook(self._hook(i, pos)))
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @staticmethod
    def _batch(x: torch.Tensor, b: int) -> torch.Tensor:
        if x.shape[0] == b:
            return x
        if b % x.shape[0]:
            raise ValueError(f"condition batch {x.shape[0]} does not divide hidden batch {b}")
        return x.repeat(b // x.shape[0], *([1] * (x.dim() - 1)))  # e.g. CFG runs cond+uncond in one batch

    def _hook(self, block_id: int, pos: int):
        def hook(module, args, output):
            if self._grid is None:  # not conditioned: pure backbone
                return output
            hidden = output
            b, length, _ = hidden.shape
            if length != self._grid[0] * self._grid[1] * self._grid[2]:
                raise ValueError(f"hidden has {length} tokens, grid {self._grid} expects "
                                 f"{self._grid[0] * self._grid[1] * self._grid[2]}")
            res = torch.zeros_like(hidden)
            if self._cond is not None:
                cond = ConditionTokens(self._batch(self._cond.tokens, b), self._cond.frame_index,
                                       self._batch(self._cond.mask, b))
                res = res + self.cross[str(block_id)](hidden, cond, self._batch(self._allowed, b))
            if self._dense_tokens is not None:
                res = res + self.dense[str(block_id)](self._batch(self._dense_tokens, b).to(hidden.dtype))
            return hidden + self.scale[pos].to(hidden.dtype) * res
        return hook
