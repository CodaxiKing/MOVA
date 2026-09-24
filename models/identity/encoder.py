"""Identity Encoder v0: N reference images (front, side, back, close-up...) -> K global identity tokens.

Contract: images (B, N, 3, H, W) in [0, 1] + view mask (B, N) -> ConditionTokens (B, K, D), frame_index = -1.
Or precomputed patch features (B, N, P, F) (e.g. DINOv2 tokens, when their weights are authorised) with
`feature_dim=F`. Pooling uses K learned queries attending over all patches of all valid views, so the output
does not depend on view order and masked views have no influence (tested). Untrained.
"""

from __future__ import annotations

import torch
from torch import nn

from models.fusion.condition import GLOBAL, ConditionTokens


class IdentityEncoder(nn.Module):
    def __init__(self, dim: int = 256, tokens: int = 8, patch: int = 16, heads: int = 4, feature_dim: int | None = None,
                 max_patches: int = 1024):
        super().__init__()
        self.dim, self.k, self.patch, self.feature_dim = dim, tokens, patch, feature_dim
        if feature_dim is None:
            self.embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        else:
            self.embed = nn.Linear(feature_dim, dim)
        self.pos = nn.Parameter(torch.zeros(1, max_patches, dim))
        self.norm = nn.LayerNorm(dim)
        self.queries = nn.Parameter(torch.randn(tokens, dim) * dim ** -0.5)
        self.pool = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def _patches(self, x: torch.Tensor) -> torch.Tensor:
        b, n = x.shape[:2]
        if self.feature_dim is None:
            p = self.embed(x.flatten(0, 1) * 2 - 1).flatten(2).transpose(1, 2)   # (B*N, P, D)
        else:
            p = self.embed(x.flatten(0, 1))
        if p.shape[1] > self.pos.shape[1]:
            raise ValueError(f"{p.shape[1]} patches per view > max_patches {self.pos.shape[1]}")
        p = p + self.pos[:, : p.shape[1]]  # same positional code for every view: views are unordered
        return p.reshape(b, n * p.shape[1], self.dim), p.shape[1]

    def forward(self, images: torch.Tensor, view_mask: torch.Tensor | None = None) -> ConditionTokens:
        b, n = images.shape[:2]
        view_mask = torch.ones(b, n, dtype=torch.bool, device=images.device) if view_mask is None else view_mask.bool()
        if not view_mask.any(1).all():
            raise ValueError("Every sample needs at least one valid reference view")
        patches, per_view = self._patches(images)
        pad = ~view_mask.repeat_interleave(per_view, 1)
        q = self.queries.expand(b, -1, -1)
        pooled, _ = self.pool(q, self.norm(patches), self.norm(patches), key_padding_mask=pad, need_weights=False)
        tokens = self.out(pooled + q)
        return ConditionTokens(tokens, torch.full((self.k,), GLOBAL, dtype=torch.long, device=images.device),
                               torch.ones(b, self.k, dtype=torch.bool, device=images.device))
