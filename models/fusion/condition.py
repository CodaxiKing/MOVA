"""Condition fusion contract shared by motion and identity paths.

A condition is a set of tokens (B, M, D) with:
  frame_index (M,) long: latent frame the token belongs to, or -1 for global tokens (identity) that every
                         video frame may attend to;
  mask (B, M) bool: True = valid token (absent streams are masked, never zero-valued "evidence").
`fuse` concatenates sources and adds a learned type embedding per source so the adapter can tell them apart.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

GLOBAL = -1


@dataclass
class ConditionTokens:
    tokens: torch.Tensor        # (B, M, D)
    frame_index: torch.Tensor   # (M,) long
    mask: torch.Tensor          # (B, M) bool

    def __post_init__(self):
        b, m, _ = self.tokens.shape
        if self.frame_index.shape != (m,) or self.mask.shape != (b, m):
            raise ValueError(f"ConditionTokens shapes: tokens {tuple(self.tokens.shape)}, "
                             f"frame_index {tuple(self.frame_index.shape)}, mask {tuple(self.mask.shape)}")

    @property
    def dim(self) -> int:
        return self.tokens.shape[-1]


class ConditionFusion(nn.Module):
    def __init__(self, dim: int, sources: tuple[str, ...] = ("motion", "identity")):
        super().__init__()
        self.sources = sources
        self.type_embed = nn.Parameter(torch.zeros(len(sources), dim))

    def forward(self, **conds: ConditionTokens | None) -> ConditionTokens:
        parts = [(self.sources.index(k), c) for k, c in conds.items() if c is not None]
        unknown = set(conds) - set(self.sources)
        if unknown or not parts:
            raise ValueError(f"Expected at least one of {self.sources}, got {sorted(conds)}")
        tokens = torch.cat([c.tokens + self.type_embed[i] for i, c in parts], 1)
        return ConditionTokens(tokens, torch.cat([c.frame_index for _, c in parts]),
                               torch.cat([c.mask for _, c in parts], 1))
